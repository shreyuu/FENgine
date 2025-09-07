import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Upload from './pages/Upload';
import Review from './pages/Review';
import Result from './pages/Result';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <h1>FENgine App</h1>
      <Router>
        <Routes>
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/review" element={<Review />} />
          <Route path="/result" element={<Result />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;