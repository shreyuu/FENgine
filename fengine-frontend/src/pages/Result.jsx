import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function Result() {
    const { state } = useLocation();
    const navigate = useNavigate();
    const [result, setResult] = useState({ fen: '', pgn: '' });

    useEffect(() => {
        if (!state) return navigate('/upload');
        // Send file and corrections to backend
        const formData = new FormData();
        formData.append('file', state.file);
        // TODO: append corrections
        axios.post('/api/ocr', formData).then(res => setResult(res.data));
    }, [state, navigate]);

    return (
        <div className="prose p-4">
            <h1>OCR Result</h1>
            <h2>FEN</h2>
            <code className="block p-2 bg-gray-200">{result.fen}</code>
            <h2>PGN</h2>
            <pre className="p-2 bg-gray-100">{result.pgn}</pre>
        </div>
    );
}
