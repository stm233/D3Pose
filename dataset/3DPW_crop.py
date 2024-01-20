import os
import cv2
import yolov7

# Load the model and set parameters
model = yolov7.load('kadirnar/yolov7-v0.1', hf_model=True)
model.conf = 0.25
model.iou = 0.45
model.classes = None


def cropandSave(model, img_path, save_path):
    img = img_path

    results = model(img)
    predictions = results.pred[0]
    boxes = predictions[:, :4]

    # Filter for 'person' class and find the leftmost person
    person_class_index = 0  # Adjust this index based on your model
    person_predictions = [box for box, category in zip(boxes, results.pred[0][:, 5]) if category == person_class_index]
    leftmost_box = min(person_predictions, key=lambda x: x[0]) if person_predictions else None

    if leftmost_box is not None:
        x1, y1, x2, y2 = map(int, leftmost_box)
        original_img = cv2.imread(img_path)
        cropped_person = original_img[y1:y2, x1:x2]

        # Save the cropped image
        cv2.imwrite(save_path, cropped_person)
        print(f'Cropped image saved to {save_path}')

        # Display the cropped image
        # cv2.imshow('Cropped Person', cropped_person)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':

    model = yolov7.load('kadirnar/yolov7-v0.1', hf_model=True)
    model.conf = 0.25
    model.iou = 0.45
    model.classes = None

    # Set the image path
    src_path = '/media/hongji/Expansion/3DPW/images'
    output_path = '/media/hongji/Expansion/3DPW/cropped_images'

    for vid in os.listdir(src_path):
        vid_path = os.path.join(src_path, vid)
        output_vid_path = os.path.join(output_path, vid)
        os.makedirs(output_vid_path)
        for img in os.listdir(vid_path):

            img_path = os.path.join(vid_path, img)
            output_img_path = os.path.join(output_vid_path, img)

            cropandSave(model, img_path, output_img_path)
