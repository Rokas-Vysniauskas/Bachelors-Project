using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using TMPro;
using Unity.Profiling;
using System.Text;

/// <summary>
/// Passive performance monitor. 
/// Automatically detects newly loaded scenes to calculate geometry and tracks hardware stats.
/// To disable profiling completely, simply Disable this GameObject.
/// </summary>
public class PerformanceOverlayController : MonoBehaviour
{
    [Header("Input Settings")]
    [Tooltip("Key to reset the FPS and Memory statistics.")]
    public Key resetKey = Key.O;

    [Header("UI References (Optional - Auto-creates if null)")]
    public Canvas mainCanvas;
    public TextMeshProUGUI statsText;

    [Header("Settings")]
    [Tooltip("How often to update the UI (in seconds).")]
    public float uiRefreshRate = 0.5f;
    [Tooltip("Number of frames to keep for 1% low calculations.")]
    public int sampleSize = 1000;

    [Header("Geometry Settings")]
    [Tooltip("If true, only counts objects with the specified tag. If false, counts ALL geometry in the additive scene.")]
    public bool useTagFilter = false;
    [Tooltip("Tag to search for geometry calculations (e.g., Destructible walls).")]
    public string targetTag = "Destructible";
    [Tooltip("How often to recalculate geometry (in seconds). Set to 0 to disable auto-refresh.")]
    public float geometryRefreshRate = 5.0f;

    // --- State ---
    private string _currentSceneName = "Waiting...";
    private float _uiTimer;
    private float _geometryTimer;
    private StringBuilder _sb = new StringBuilder(500);

    // --- Profiler Recorders ---
    private ProfilerRecorder _totalReservedMemoryRecorder;
    private ProfilerRecorder _gcReservedMemoryRecorder;
    private ProfilerRecorder _textureMemoryRecorder;
    private ProfilerRecorder _mainThreadTimeRecorder;
    private ProfilerRecorder _gpuFrameTimeRecorder;

    // --- Metrics Storage (Zero Allocation Circular Buffer) ---
    private float[] _frameTimes;
    private int _frameIndex = 0;
    private int _frameCount = 0;

    private long _currentSceneTriangles = 0;
    private long _currentSceneVertices = 0;

    // RAM/VRAM Stats (tracking min/max/avg over the session)
    private long _ramMin = long.MaxValue, _ramMax = 0;
    private double _ramAvgSum = 0; private int _ramSampleCount = 0;

    private long _vramMin = long.MaxValue, _vramMax = 0;
    private double _vramAvgSum = 0; private int _vramSampleCount = 0;

    void OnEnable()
    {
        _totalReservedMemoryRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "Total Reserved Memory");
        _gcReservedMemoryRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "GC Reserved Memory");
        _textureMemoryRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Memory, "Texture Memory");
        _mainThreadTimeRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Internal, "Main Thread", 15);
        _gpuFrameTimeRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Gpu Frame Time", 15);

        SceneManager.sceneLoaded += OnSceneLoaded;

        if (mainCanvas != null) mainCanvas.gameObject.SetActive(true);

        if (SceneManager.GetActiveScene().isLoaded)
        {
            _currentSceneName = SceneManager.GetActiveScene().name;
            CalculateDestructibleGeometry();
        }
    }

    void OnDisable()
    {
        _totalReservedMemoryRecorder.Dispose();
        _gcReservedMemoryRecorder.Dispose();
        _textureMemoryRecorder.Dispose();
        _mainThreadTimeRecorder.Dispose();
        _gpuFrameTimeRecorder.Dispose();

        SceneManager.sceneLoaded -= OnSceneLoaded;

        if (mainCanvas != null) mainCanvas.gameObject.SetActive(false);
    }

    void OnDestroy()
    {
        if (mainCanvas != null) Destroy(mainCanvas.gameObject);
    }

    void Start()
    {
        EnsureUI();
        _frameTimes = new float[sampleSize];
    }

    void Update()
    {
        // 0. Handle Input (Reset Stats)
        if (Keyboard.current != null && Keyboard.current[resetKey].wasPressedThisFrame)
        {
            ResetStats();
        }

        // 1. Collect Frame Metrics
        float dt = Time.unscaledDeltaTime;
        float dtMs = dt * 1000.0f;

        // Write to array and wrap around for zero-allocation
        if (_frameTimes != null && _frameTimes.Length > 0)
        {
            _frameTimes[_frameIndex] = dtMs;
            _frameIndex = (_frameIndex + 1) % sampleSize;
            if (_frameCount < sampleSize) _frameCount++;
        }

        // 2. Update Memory Stats
        UpdateMemoryStats();

        // 3. Geometry Refresh Logic
        if (geometryRefreshRate > 0)
        {
            _geometryTimer += Time.unscaledDeltaTime;
            if (_geometryTimer >= geometryRefreshRate)
            {
                CalculateDestructibleGeometry();
                _geometryTimer = 0;
            }
        }

        // 4. Refresh UI
        _uiTimer += Time.unscaledDeltaTime;
        if (_uiTimer >= uiRefreshRate)
        {
            UpdateUI(dtMs);
            _uiTimer = 0;
        }
    }

    // --- State Management ---

    public void ResetStats()
    {
        _frameIndex = 0;
        _frameCount = 0;

        _ramMin = long.MaxValue; _ramMax = 0;
        _ramAvgSum = 0; _ramSampleCount = 0;

        _vramMin = long.MaxValue; _vramMax = 0;
        _vramAvgSum = 0; _vramSampleCount = 0;

        _uiTimer = uiRefreshRate;
    }

    // --- Event Handling ---

    void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        _currentSceneName = scene.name;
        CalculateDestructibleGeometry();
        _geometryTimer = 0;
    }

    // --- Geometry Calculation ---

    void CalculateDestructibleGeometry()
    {
        _currentSceneTriangles = 0;
        _currentSceneVertices = 0;

        if (useTagFilter)
        {
            if (string.IsNullOrEmpty(targetTag)) return;

            GameObject[] targets;
            try
            {
                targets = GameObject.FindGameObjectsWithTag(targetTag);
            }
            catch (UnityException)
            {
                return;
            }

            foreach (var obj in targets)
            {
                if (obj.scene.name != _currentSceneName && _currentSceneName != "Waiting...") continue;
                CountObjectGeometry(obj);
            }
        }
        else
        {
            if (_currentSceneName == "Waiting...") return;

            Scene scene = SceneManager.GetSceneByName(_currentSceneName);
            if (!scene.IsValid() || !scene.isLoaded) return;

            GameObject[] roots = scene.GetRootGameObjects();
            foreach (var root in roots)
            {
                CountObjectGeometry(root);
            }
        }
    }

    void CountObjectGeometry(GameObject obj)
    {
        MeshFilter[] meshFilters = obj.GetComponentsInChildren<MeshFilter>(false);
        foreach (var mf in meshFilters)
        {
            if (mf.sharedMesh != null)
            {
                for (int i = 0; i < mf.sharedMesh.subMeshCount; i++)
                {
                    _currentSceneTriangles += (mf.sharedMesh.GetIndexCount(i) / 3);
                }
                _currentSceneVertices += mf.sharedMesh.vertexCount;
            }
        }

        SkinnedMeshRenderer[] skinnedMeshes = obj.GetComponentsInChildren<SkinnedMeshRenderer>(false);
        foreach (var smr in skinnedMeshes)
        {
            if (smr.sharedMesh != null)
            {
                for (int i = 0; i < smr.sharedMesh.subMeshCount; i++)
                {
                    _currentSceneTriangles += (smr.sharedMesh.GetIndexCount(i) / 3);
                }
                _currentSceneVertices += smr.sharedMesh.vertexCount;
            }
        }
    }

    // --- Metrics Logic ---

    void UpdateMemoryStats()
    {
        if (!_totalReservedMemoryRecorder.Valid) return;

        long currentRam = _totalReservedMemoryRecorder.LastValue / (1024 * 1024);
        long currentVram = _textureMemoryRecorder.LastValue / (1024 * 1024);

        if (currentRam < _ramMin) _ramMin = currentRam;
        if (currentRam > _ramMax) _ramMax = currentRam;
        _ramAvgSum += currentRam;
        _ramSampleCount++;

        if (currentVram < _vramMin) _vramMin = currentVram;
        if (currentVram > _vramMax) _vramMax = currentVram;
        _vramAvgSum += currentVram;
        _vramSampleCount++;
    }

    void UpdateUI(float currentDtMs)
    {
        if (_frameCount == 0) return;

        // Create a temporary array to sort so we don't mess up the circular buffer
        float[] sortedTimes = new float[_frameCount];
        Array.Copy(_frameTimes, sortedTimes, _frameCount);
        Array.Sort(sortedTimes); // In-place sort, zero garbage allocation

        // --- FPS Calculations ---
        float currentFps = 1000.0f / (currentDtMs > 0 ? currentDtMs : 0.001f);

        float sumAll = 0f;
        for (int i = 0; i < _frameCount; i++) sumAll += sortedTimes[i];
        float avgFrameTime = sumAll / _frameCount;

        // 1% Low FPS = AVERAGE of the slowest 1% of frames
        int onePercentCount = Mathf.Max(1, Mathf.FloorToInt(_frameCount * 0.01f));
        float sumWorst = 0f;

        // The slowest frames are at the end of the ascending sorted array
        for (int i = _frameCount - onePercentCount; i < _frameCount; i++)
        {
            sumWorst += sortedTimes[i];
        }
        float frameTime1PercentLow = sumWorst / onePercentCount;

        // Frame Time Min (Fastest frame)
        float frameTimeMin = sortedTimes[0];

        // FPS Metrics
        float fpsMax = 1000.0f / (frameTimeMin > 0 ? frameTimeMin : 0.001f);
        float fpsAvg = 1000.0f / (avgFrameTime > 0 ? avgFrameTime : 0.001f);
        float fps1PercentLow = 1000.0f / (frameTime1PercentLow > 0 ? frameTime1PercentLow : 0.001f);

        // --- Hardware Usage ---
        double cpuTimeMs = _mainThreadTimeRecorder.Valid ? _mainThreadTimeRecorder.LastValue * (1e-6f) : 0;
        double gpuTimeMs = _gpuFrameTimeRecorder.Valid ? _gpuFrameTimeRecorder.LastValue * (1e-6f) : 0;

        float targetMs = 16.66f;
        float cpuLoad = (float)(cpuTimeMs / targetMs) * 100f;
        float gpuLoad = (float)(gpuTimeMs / targetMs) * 100f;

        long ramAvg = _ramSampleCount > 0 ? (long)(_ramAvgSum / _ramSampleCount) : 0;
        long vramAvg = _vramSampleCount > 0 ? (long)(_vramAvgSum / _vramSampleCount) : 0;

        // --- Text Generation ---
        _sb.Clear();
        _sb.Append($"<b><size=120%>{_currentSceneName}</size></b>\n\n");

        _sb.Append("<b>FPS:</b>\n");
        _sb.Append($"  Max: <color=green>{fpsMax:F0}</color> | Curr: <color=white>{currentFps:F0}</color> | Avg: <color=yellow>{fpsAvg:F0}</color> | 1% Low: <color=red>{fps1PercentLow:F0}</color>\n");

        _sb.Append("<b>Frame Time (ms):</b>\n");
        _sb.Append($"  Min: {frameTimeMin:F2} | Curr: {currentDtMs:F2} | Avg: {avgFrameTime:F2} | Max (1%): {frameTime1PercentLow:F2}\n");

        _sb.Append($"\n<b>Geometry ({(useTagFilter ? $"Tag: {targetTag}" : "Active Only")}):</b>\n");
        _sb.Append($"  Tris: {_currentSceneTriangles:N0}\n");
        _sb.Append($"  Verts: {_currentSceneVertices:N0}\n");

        _sb.Append("\n<b>Hardware Load (~60fps):</b>\n");
        _sb.Append($"  CPU: {cpuTimeMs:F2}ms ({cpuLoad:F0}%)\n");
        _sb.Append($"  GPU: {gpuTimeMs:F2}ms ({gpuLoad:F0}%)\n");

        _sb.Append("\n<b>Memory (MB):</b>\n");
        _sb.Append($"  RAM:  Min: {_ramMin} | Avg: {ramAvg} | Max: {_ramMax}\n");
        _sb.Append($"  VRAM: Min: {_vramMin} | Avg: {vramAvg} | Max: {_vramMax} (Tex)\n");

        if (statsText != null)
        {
            statsText.text = _sb.ToString();
        }
    }

    // --- Helpers ---

    private void EnsureUI()
    {
        if (mainCanvas == null)
        {
            GameObject canvasObj = new GameObject("PerformanceCanvas");
            mainCanvas = canvasObj.AddComponent<Canvas>();
            mainCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
            mainCanvas.sortingOrder = 999;
            canvasObj.AddComponent<CanvasScaler>();
            canvasObj.AddComponent<GraphicRaycaster>();
            DontDestroyOnLoad(canvasObj);
        }

        if (statsText == null)
        {
            GameObject bgObj = new GameObject("StatsBackground");
            bgObj.transform.SetParent(mainCanvas.transform, false);

            Image bg = bgObj.AddComponent<Image>();
            bg.color = new Color(0, 0, 0, 0.5f);

            VerticalLayoutGroup layout = bgObj.AddComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(10, 10, 10, 10);
            layout.childControlWidth = true;
            layout.childControlHeight = true;
            layout.childForceExpandWidth = false;
            layout.childForceExpandHeight = false;

            ContentSizeFitter csf = bgObj.AddComponent<ContentSizeFitter>();
            csf.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
            csf.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

            RectTransform bgRt = bgObj.GetComponent<RectTransform>();
            bgRt.anchorMin = new Vector2(0, 1);
            bgRt.anchorMax = new Vector2(0, 1);
            bgRt.pivot = new Vector2(0, 1);
            bgRt.anchoredPosition = new Vector2(10, -10);

            GameObject textObj = new GameObject("StatsText");
            textObj.transform.SetParent(bgObj.transform, false);

            statsText = textObj.AddComponent<TextMeshProUGUI>();
            statsText.fontSize = 18;
            statsText.color = Color.white;
            statsText.alignment = TextAlignmentOptions.TopLeft;
        }
    }
}