import React, { useState } from 'react';
import { MapContainer, TileLayer, Polyline } from 'react-leaflet';
import './App.css';

function App() {
  const chicago_coords = [41.8781, -87.6298];
  const [originText, setOriginText] = useState("");
  const [destText, setDestText] = useState("");
  const [routes, setRoutes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFindRoutes = async () => {
    // If input is empty, fall back to the placeholder value for easy testing
    const finalOrigin = originText.trim() || "41.8827, -87.6233";
    const finalDest = destText.trim() || "41.8750, -87.6350";
    
    try {
      const [oLatStr, oLonStr] = finalOrigin.split(",");
      const [dLatStr, dLonStr] = finalDest.split(",");
      
      const payload = {
        origin_lat: parseFloat(oLatStr.trim()),
        origin_lon: parseFloat(oLonStr.trim()),
        dest_lat: parseFloat(dLatStr.trim()),
        dest_lon: parseFloat(dLonStr.trim())
      };

      setLoading(true);
      setError(null);

      const response = await fetch("http://localhost:8000/routes", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error("Failed to fetch");
      }

      const data = await response.json();
      setRoutes(data.routes || []);
      console.log("Fetched routes:", data.routes);
      
    } catch (err) {
      console.error("API error:", err);
      setError("Error fetching routes. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  const getColor = (label) => {
    if (label.includes("safety")) return "#22c55e"; // green
    if (label.includes("speed")) return "#ef4444"; // red
    return "#f97316"; // orange (balanced)
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <h1>SafeStride</h1>
        <p>Safer routes for everyone</p>

        <div style={{ marginTop: '20px' }}>
          <div style={{ marginBottom: '10px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Origin:</label>
            <input 
              type="text" 
              placeholder="41.8827, -87.6233" 
              value={originText}
              onChange={(e) => setOriginText(e.target.value)}
              style={{ width: '100%', padding: '8px', boxSizing: 'border-box' }}
            />
          </div>
          
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>Destination:</label>
            <input 
              type="text" 
              placeholder="41.8750, -87.6350" 
              value={destText}
              onChange={(e) => setDestText(e.target.value)}
              style={{ width: '100%', padding: '8px', boxSizing: 'border-box' }}
            />
          </div>
          
          <button 
            onClick={handleFindRoutes}
            disabled={loading}
            style={{ width: '100%', padding: '10px', backgroundColor: loading ? '#ccc' : '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 'bold' }}
          >
            Find Safe Routes
          </button>
          
          {loading && <p style={{ marginTop: '10px', color: '#007bff', fontSize: '14px' }}>Finding routes...</p>}
          {error && <p style={{ marginTop: '10px', color: 'red', fontSize: '14px' }}>{error}</p>}
        </div>
      </div>
      <div className="map-container">
        <MapContainer center={chicago_coords} zoom={13} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {routes && routes.map((route, i) => {
            if (!route.geometry || route.geometry.length === 0) return null;
            return (
              <Polyline 
                key={i} 
                positions={route.geometry} 
                pathOptions={{ color: getColor(route.label), weight: 4, opacity: 0.8 }} 
              />
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
