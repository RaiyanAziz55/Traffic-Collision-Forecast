import { useCallback, useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';
const TILE_URL = import.meta.env.VITE_TILE_URL ?? 'http://localhost:3000/ottawa_risk_baseline';
const TILE_SCAN_PADDING = 0;

function getTodayDateString() {
  return new Date().toLocaleDateString('en-CA');
}

function buildLocalDate(dateString) {
  return new Date(`${dateString}T12:00:00`);
}

function describeWeather(temp, prcp, snow) {
  if (snow > 0 || (prcp > 0 && temp < 0)) return 'Snowing / Freezing';
  if (prcp > 0) return 'Raining / Wet';
  return 'Clear / Dry';
}

function getRiskBand(score, thresholds) {
  if (score >= thresholds.red) return 'High';
  if (score >= thresholds.yellow) return 'Moderate';
  return 'Lower';
}

function lngLatToTile(lng, lat, zoom) {
  const scale = 2 ** zoom;
  const x = Math.floor(((lng + 180) / 360) * scale);
  const latRad = (lat * Math.PI) / 180;
  const y = Math.floor(
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * scale
  );

  return { x, y, z: zoom };
}

function tileToBounds(x, y, z) {
  const scale = 2 ** z;
  const west = (x / scale) * 360 - 180;
  const east = ((x + 1) / scale) * 360 - 180;

  const northRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / scale)));
  const southRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 1)) / scale)));

  return {
    west,
    east,
    north: (northRad * 180) / Math.PI,
    south: (southRad * 180) / Math.PI,
  };
}

function isLngLatInBounds(lngLat, bounds) {
  return (
    lngLat.lng >= bounds.west &&
    lngLat.lng <= bounds.east &&
    lngLat.lat >= bounds.south &&
    lngLat.lat <= bounds.north
  );
}

function App() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const activeScoresRef = useRef({});
  const loadedAreasRef = useRef({});
  const riskThresholdsRef = useRef({ yellow: 0.35, red: 0.55 });
  const weatherRef = useRef({
    temp: 15,
    prcp: 0,
    snow: 0,
    condition: 'Loading...',
    mode: 'live',
    activeDate: getTodayDateString(),
    label: 'Live Ottawa Weather',
  });
  
  const [panelData, setPanelData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [weatherTracker, setWeatherTracker] = useState({
    temp: null,
    prcp: 0,
    snow: 0,
    condition: 'Loading...',
    mode: 'live',
    activeDate: getTodayDateString(),
    label: 'Live Ottawa Weather',
  });
  const [weatherDateInput, setWeatherDateInput] = useState(getTodayDateString());
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [, setActiveScores] = useState({});

  const syncWeatherState = useCallback((nextWeather) => {
    weatherRef.current = nextWeather;
    setWeatherTracker(nextWeather);
  }, []);

  const loadLiveWeather = useCallback(async () => {
    setWeatherLoading(true);
    try {
      const weatherRes = await fetch('https://api.open-meteo.com/v1/forecast?latitude=45.4215&longitude=-75.6972&current=temperature_2m,rain,snowfall');
      const weatherData = await weatherRes.json();
      const current = weatherData.current;
      const nextWeather = {
        temp: current.temperature_2m,
        prcp: current.rain,
        snow: current.snowfall,
        condition: describeWeather(current.temperature_2m, current.rain, current.snowfall),
        mode: 'live',
        activeDate: getTodayDateString(),
        label: 'Live Ottawa Weather',
      };
      setWeatherDateInput(nextWeather.activeDate);
      syncWeatherState(nextWeather);
    } catch {
      console.error('Weather API failed');
    } finally {
      setWeatherLoading(false);
    }
  }, [syncWeatherState]);

  const loadHistoricalWeather = useCallback(async (dateString) => {
    setWeatherLoading(true);
    try {
      const archiveUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=45.4215&longitude=-75.6972&start_date=${dateString}&end_date=${dateString}&daily=temperature_2m_mean,precipitation_sum,snowfall_sum&timezone=America%2FToronto`;
      const weatherRes = await fetch(archiveUrl);
      const weatherData = await weatherRes.json();
      const daily = weatherData.daily;

      if (!daily || !daily.time?.length) {
        throw new Error('Historical weather unavailable');
      }

      const temp = daily.temperature_2m_mean?.[0] ?? 0;
      const prcp = daily.precipitation_sum?.[0] ?? 0;
      const snow = daily.snowfall_sum?.[0] ?? 0;

      syncWeatherState({
        temp,
        prcp,
        snow,
        condition: describeWeather(temp, prcp, snow),
        mode: 'historical',
        activeDate: dateString,
        label: `Ottawa Weather • ${dateString}`,
      });
    } catch {
      console.error('Historical weather API failed');
    } finally {
      setWeatherLoading(false);
    }
  }, [syncWeatherState]);

  const handleWeatherDateChange = async (event) => {
    const nextDate = event.target.value;
    setWeatherDateInput(nextDate);
    if (!nextDate) return;

    if (nextDate === getTodayDateString()) {
      await loadLiveWeather();
      return;
    }

    await loadHistoricalWeather(nextDate);
  };

  useEffect(() => {
    // 1. Initialize Map synchronously FIRST so React Strict Mode catches it immediately
    if (map.current) return;
    
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [-75.6972, 45.4215],
      zoom: 14
    });

    const fetchModelMetadata = async () => {
      try {
        const metadataRes = await fetch(`${API_BASE_URL}/model-metadata`);
        if (!metadataRes.ok) return;

        const metadata = await metadataRes.json();
        if (metadata?.suggested_thresholds?.yellow && metadata?.suggested_thresholds?.red) {
          riskThresholdsRef.current = metadata.suggested_thresholds;
        }
      } catch {
        console.error('Model metadata fetch failed');
      }
    };

    // 2. Fetch Weather and model metadata asynchronously AFTER map initialization
    loadLiveWeather();
    fetchModelMetadata();

    map.current.on('load', () => {
      // 3. Safety check to prevent MapLibre from crashing on hot-reloads
      if (map.current.getSource('ottawa-roads')) return;

      map.current.addSource('ottawa-roads', {
        type: 'vector',
        url: TILE_URL
      });

      // Layer 1: The Neutral Grey Baseline
      map.current.addLayer({
        id: 'road-baseline',
        type: 'line',
        source: 'ottawa-roads',
        'source-layer': 'ottawa_risk_baseline',
        paint: {
          'line-width': ['interpolate', ['linear'], ['zoom'], 10, 1, 15, 3],
          'line-color': '#333' 
        }
      });

      // Layer 2: The Colored Neighborhoods (Dynamic)
      map.current.addLayer({
        id: 'road-colored',
        type: 'line',
        source: 'ottawa-roads',
        'source-layer': 'ottawa_risk_baseline',
        paint: {
          'line-width': ['interpolate', ['linear'], ['zoom'], 10, 3, 15, 6],
          'line-color': ['match', ['concat', ['to-string', ['get', 'u']], '_', ['to-string', ['get', 'v']]], 'fallback_id', '#555', 'rgba(0,0,0,0)']
        }
      });

      const syncRoadColors = () => {
        const mergedScores = Object.values(loadedAreasRef.current).reduce((acc, area) => {
          return { ...acc, ...area.scores };
        }, {});

        activeScoresRef.current = mergedScores;
        setActiveScores(mergedScores);

        if (Object.keys(mergedScores).length === 0) {
          map.current.setPaintProperty('road-colored', 'line-color', 'rgba(0,0,0,0)');
          return;
        }

        const matchExpression = [
          'match',
          ['concat', ['to-string', ['get', 'u']], '_', ['to-string', ['get', 'v']]]
        ];

        const { yellow, red } = riskThresholdsRef.current;
        for (const [id, score] of Object.entries(mergedScores)) {
          let color = '#2ecc71'; // Green
          if (score > yellow) color = '#f1c40f'; // Yellow
          if (score > red) color = '#e74c3c'; // Red
          matchExpression.push(id, color);
        }

        matchExpression.push('rgba(0,0,0,0)');

        map.current.setPaintProperty('road-colored', 'line-color', matchExpression);
      };

      // THE CLICK EVENT
      map.current.on('click', async (e) => {
        const clickedRoads = map.current.queryRenderedFeatures(e.point, { layers: ['road-colored', 'road-baseline'] });
        const clickedFeature = clickedRoads[0];

        if (clickedFeature) {
          const segmentId = `${clickedFeature.properties.u}_${clickedFeature.properties.v}`;
          const existingScore = activeScoresRef.current[segmentId];

          // --- SCENARIO A: ROAD IS ALREADY SCORED -> TRIGGER LLM ---
          if (existingScore !== undefined) {
            const currentWeather = weatherRef.current;
            setPanelData({
              properties: clickedFeature.properties,
              explanation: '',
              riskMultiplier: existingScore,
              weatherSnapshot: currentWeather,
            });
            setLoading(true);

            try {
              const res = await fetch(`${API_BASE_URL}/insight`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  name: clickedFeature.properties.name,
                  highway: clickedFeature.properties.highway,
                  historical_count: clickedFeature.properties.historical_count,
                  temp: currentWeather.temp,
                  prcp: currentWeather.prcp,
                  snow: currentWeather.snow,
                  risk_multiplier: existingScore
                })
              });
              const data = await res.json();
              setPanelData((previous) => previous ? {
                ...previous,
                explanation: data.explanation,
              } : previous);
            } catch (err) { console.error(err); }
            setLoading(false);
            return;
          }
        }

        // --- SCENARIO B: CLICK ANYWHERE -> TOGGLE THE SURROUNDING TILE AREA ---
        const loadedAreaEntry = Object.entries(loadedAreasRef.current).find(([, area]) =>
          isLngLatInBounds(e.lngLat, area.bounds)
        );

        if (loadedAreaEntry) {
          const [existingAreaKey] = loadedAreaEntry;
          delete loadedAreasRef.current[existingAreaKey];
          setPanelData(null);
          syncRoadColors();
          return;
        }

        const tileZoom = Math.max(0, Math.floor(map.current.getZoom()));
        const tile = lngLatToTile(e.lngLat.lng, e.lngLat.lat, tileZoom);
        const areaKey = `${tile.z}/${tile.x}/${tile.y}`;
        const bounds = tileToBounds(tile.x, tile.y, tile.z);
        const northWest = map.current.project([bounds.west, bounds.north]);
        const southEast = map.current.project([bounds.east, bounds.south]);
        const bbox = [
          [northWest.x - TILE_SCAN_PADDING, northWest.y - TILE_SCAN_PADDING],
          [southEast.x + TILE_SCAN_PADDING, southEast.y + TILE_SCAN_PADDING]
        ];
        
        const features = map.current.queryRenderedFeatures(bbox, { layers: ['road-baseline'] });
        const uniqueRoads = [];
        const seen = new Set();
        
        features.forEach(f => {
          const id = `${f.properties.u}_${f.properties.v}`;
          if (!seen.has(id)) {
            seen.add(id);
            uniqueRoads.push(f.properties);
          }
        });

        if (uniqueRoads.length === 0) return;

        try {
          const activeDate = weatherRef.current.activeDate || getTodayDateString();
          const selectedDate = buildLocalDate(activeDate);
          const pythonStyleDayOfWeek = (selectedDate.getDay() + 6) % 7;
          const res = await fetch(`${API_BASE_URL}/predict-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              roads: uniqueRoads,
              temp: weatherRef.current.temp,
              prcp: weatherRef.current.prcp,
              snow: weatherRef.current.snow,
              month: selectedDate.getMonth() + 1,
              day_of_week: pythonStyleDayOfWeek
            })
          });
          
          const data = await res.json();
          const newScores = data.scores;
          loadedAreasRef.current[areaKey] = {
            bounds,
            scores: newScores,
          };
          syncRoadColors();

        } catch (error) { console.error(error); }
      });
      
      // Make cursor a pointer when hovering over any road
      map.current.on('mouseenter', 'road-baseline', () => map.current.getCanvas().style.cursor = 'pointer');
      map.current.on('mouseleave', 'road-baseline', () => map.current.getCanvas().style.cursor = '');
    });
  }, [loadLiveWeather]);

  return (
    <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0, position: 'relative' }}>
      <div ref={mapContainer} style={{ width: '100%', height: '100%' }} />
      
      <div style={{
        position: 'absolute', top: 20, left: 20, 
        background: '#1e1e1e', color: 'white', padding: '10px 15px', 
        borderRadius: '8px', zIndex: 1, borderLeft: '4px solid #3498db'
      }}>
        <div style={{ fontSize: '12px', color: '#aaa', textTransform: 'uppercase' }}>{weatherTracker.label}</div>
        <div style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '5px' }}>
          {weatherTracker.temp !== null ? `${weatherTracker.temp}°C` : '--'} • {weatherTracker.condition}
        </div>
        <div style={{ fontSize: '11px', color: '#777', marginTop: '5px' }}>
          Rain {weatherTracker.prcp} mm • Snow {weatherTracker.snow} mm
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '10px' }}>
          <input
            type="date"
            value={weatherDateInput}
            max={getTodayDateString()}
            onChange={handleWeatherDateChange}
            style={{
              background: '#111',
              color: 'white',
              border: '1px solid #333',
              borderRadius: '6px',
              padding: '6px 8px',
              fontSize: '12px'
            }}
          />
          <button
            onClick={() => loadLiveWeather()}
            style={{
              background: weatherTracker.mode === 'live' ? '#3498db' : '#222',
              color: 'white',
              border: '1px solid #333',
              borderRadius: '6px',
              padding: '6px 10px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            Live
          </button>
        </div>
        <div style={{ fontSize: '11px', color: '#777', marginTop: '8px' }}>
          {weatherLoading ? 'Updating weather...' : 'Click map to scan area'}
        </div>
      </div>

      {(panelData || loading) && (
        <div style={{
          position: 'absolute', top: 20, right: 20, width: '320px',
          background: '#1e1e1e', color: 'white', padding: '20px', 
          borderRadius: '8px', zIndex: 1, boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #333', paddingBottom: '10px', marginBottom: '10px' }}>
            <h3 style={{ margin: 0 }}>AI Risk Assessment</h3>
            <button onClick={() => setPanelData(null)} style={{ background: 'transparent', color: '#888', border: 'none', cursor: 'pointer', fontSize: '16px' }}>✖</button>
          </div>

          {panelData && (
            <>
              <div style={{ marginBottom: '15px', fontSize: '14px', color: '#ccc' }}>
                <div><strong>Road:</strong> {panelData.properties.name || 'Unknown Segment'}</div>
                <div><strong>Type:</strong> {panelData.properties.highway}</div>
                <div><strong>Historical collisions:</strong> {Math.round(Number(panelData.properties.historical_count || 0))}</div>
                <div><strong>Risk score:</strong> {(panelData.riskMultiplier * 100).toFixed(2)}%</div>
                <div><strong>Risk band:</strong> {getRiskBand(panelData.riskMultiplier, riskThresholdsRef.current)}</div>
                <div><strong>Weather date:</strong> {panelData.weatherSnapshot?.activeDate}</div>
                <div><strong>Weather:</strong> {panelData.weatherSnapshot?.temp}°C, rain {panelData.weatherSnapshot?.prcp} mm, snow {panelData.weatherSnapshot?.snow} mm</div>
                <div style={{ marginTop: '6px', color: '#9a9a9a' }}>
                  Segment score is based on this road segment's historical pattern plus the selected weather, not raw collision count alone.
                </div>
              </div>

              <div style={{ borderTop: '1px solid #333', paddingTop: '12px' }}>
                <div style={{ fontSize: '12px', color: '#888', textTransform: 'uppercase', marginBottom: '8px' }}>AI Assessment</div>
                {loading ? (
                  <p style={{ color: '#aaa', fontStyle: 'italic', margin: 0 }}>Generating AI insight via Ollama...</p>
                ) : (
                  <p style={{ lineHeight: '1.5', fontSize: '14px', margin: 0, color: '#e0e0e0' }}>
                    {panelData.explanation || 'No AI assessment available yet.'}
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
