import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function Upload() {
    const [file, setFile] = useState(null);
    const navigate = useNavigate();

    const handleSubmit = () => {
        if (!file) return;
        navigate('/review', { state: { file } });
    };

    return (
        <div className="flex flex-col items-center justify-center p-8">
            <h1 className="text-2xl font-semibold mb-4">Upload Chess Board Image</h1>
            <input
                type="file"
                accept="image/*"
                onChange={e => setFile(e.target.files[0])}
                className="mb-4"
            />
            <button
                onClick={handleSubmit}
                className="px-6 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
                disabled={!file}
            >
                Next: Review
            </button>
        </div>
    );
}
