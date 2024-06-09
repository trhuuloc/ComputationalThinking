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
                    context.strokeStyle = 'red';
                    context.lineWidth = 2;
                    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
                });
            };

            image.src = previewImage;
        }
    }, [previewImage, faces]);

    return (
        <div>
            <input type="file" accept="image/*" onChange={handleImageChange} />
            <button className='button-30' id='Detect' onClick={detectFaces}>Detect Faces</button>
            {previewImage && <canvas ref={canvasRef} style={{ display: 'block', maxWidth: '100%' }} />}
            <div>
                {faces.map((face, index) => (
                    <div key={index}>
                        <img src={`data:image/jpeg;base64,${face.cropped_image}`} alt={`Face ${index}`} />
                        <p>{face.classifier}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default App;
