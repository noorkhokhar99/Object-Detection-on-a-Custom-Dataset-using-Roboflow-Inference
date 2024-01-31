import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

inference.Stream(
    source="IMG_0270.MOV", # or rtsp stream or camera id
    model="logistics-sz9jr/2",

    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    
    on_prediction=lambda predictions, image: (
        print(predictions), # now hold up your hand: 🪨 📄 ✂️
        
        cv2.imshow(
            "Prediction", 
            annotator.annotate(
                scene=image, 
                detections=sv.Detections.from_roboflow(predictions)
            )
        ),
        cv2.waitKey(1)
    )
)



