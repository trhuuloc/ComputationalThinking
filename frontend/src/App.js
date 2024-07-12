import React, { useState, useRef, useEffect } from 'react';

function App() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [previewImage, setPreviewImage] = useState(null);
    const [faces, setFaces] = useState([]);
    const canvasRef = useRef(null);

    const handleImageChange = (event) => {
        const file = event.target.files[0];
        if (file != null) {
            setSelectedImage(file);
            setPreviewImage(URL.createObjectURL(file));
        }
    };

    const detectFaces = async () => {
        if (!selectedImage) {
            alert('Please select an image first!');
            return;
        }

        const formData = new FormData();
        formData.append('image', selectedImage);

        try {
            const response = await fetch('http://127.0.0.1:5000/detect', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();
            setFaces(result.faces);
            console.log(result.time);
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to detect faces. Please try again.');
        }
    };

    useEffect(() => {
        if (previewImage && faces.length > 0) {
            const canvas = canvasRef.current;
            const context = canvas.getContext('2d');
            const image = new Image();

            image.onload = () => {
                canvas.width = image.width;
                canvas.height = image.height;
                context.drawImage(image, 0, 0, image.width, image.height);

                faces.forEach((face) => {
                    const [x1, y1, x2, y2] = face.box;
                    const boxWidth = x2 - x1;
                    const fontSize = boxWidth * 0.15;

                    context.strokeStyle = 'red';
                    context.lineWidth = 2;
                    context.strokeRect(x1, y1, boxWidth, y2 - y1);

                    context.font = `${fontSize}px Arial`;
                    context.fillStyle = 'red';
                    context.fillText(`Conf: ${face.confidence.toFixed(2)}`, x1, y1 - 2);
                });
            };

            image.src = previewImage;
        }
    }, [previewImage, faces]);

    return (
        <div className="min-h-screen flex flex-col items-center bg-gradient-to-r from-black to-gray-400 p-5">
            <h1 className="text-[100px] text-white font-bold mt-10 mb-5 animate-pulse">
                Fake Face Detection
            </h1>
            <div className="flex w-full justify-center items-start mt-5">
                <div className="flex flex-col items-center">
                    <input 
                        type="file" 
                        accept="image/*" 
                        onChange={handleImageChange} 
                        className="mb-4 px-4 py-2 text-white text-[30px] border border-gray-300 rounded-lg cursor-pointer shadow-lg transition-transform transform hover:scale-105"
                    />
                    {previewImage && (
                        <div className="mb-4">
                            <canvas ref={canvasRef} className="max-w-full border border-gray-300 rounded-lg shadow-lg"></canvas>
                        </div>
                    )}
                </div>
                <button 
                    className="mx-10 my-6 px-6 py-5 bg-gradient-to-r w-[500px] text-[50px] from-white to-gray-300 text-black rounded-lg shadow-lg hover:shadow-xl transition-transform transform hover:scale-105"
                    onClick={detectFaces}
                >
                    Detect Faces
                </button>
                <div className="flex flex-col items-center">
                    <div className="flex justify-center w-full">
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                            {faces.map((face, index) => (
                                <div key={index} className="flex flex-col items-center">
                                    <img 
                                        src={`data:image/jpeg;base64,${face.cropped_image}`} 
                                        alt={`Face ${index}`} 
                                        className="w-48 h-48 object-cover border border-gray-300 rounded-lg shadow-lg"
                                    />
                                    <p className={`mt-2 text-[40px] font-bold ${face.classifier === 'fake' ? 'text-red-500' : 'text-green-400'}`}>
                                        {face.classifier}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
