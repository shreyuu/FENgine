import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';

export default function Review() {
    const { state } = useLocation();
    const navigate = useNavigate();
    const canvasRef = useRef();
    const [_image, setImage] = useState(null);
    const [corrections, _setCorrections] = useState({});

    useEffect(() => {
        if (!state?.file) return navigate('/upload');
        const img = new Image();
        img.src = URL.createObjectURL(state.file);
        img.onload = () => {
            setImage(img);
            const ctx = canvasRef.current.getContext('2d');
            canvasRef.current.width = img.width;
            canvasRef.current.height = img.height;
            ctx.drawImage(img, 0, 0);
            // TODO: draw grid overlay
        };
    }, [state, navigate]);

    const handleConfirm = () => {
        // Pass both the file and any corrections to the result page
        navigate('/result', {
            state: {
                file: state.file,
                corrections: corrections
            }
        });
    };

    return (
        <div className="p-4">
            <h1 className="text-xl font-semibold mb-2">Review Detected Board</h1>
            <canvas ref={canvasRef} className="border" />
            <button
                onClick={handleConfirm}
                className="mt-4 px-4 py-2 bg-green-600 text-white rounded"
            >
                Confirm and Generate FEN/PGN
            </button>
        </div>
    );
}
