import cv2
import face_recognition
import os
import numpy as np
import streamlit as st


def create_dataset(name, reg_no, max_images=20):
    """
    Dataset collection with Streamlit UI feedback
    """

    dataset_path = "data/students"
    person_path = os.path.join(dataset_path, f"{reg_no}_{name}")
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    status_box = st.empty()
    progress_bar = st.progress(0)

    status_box.info("📸 Dataset collection started | Press **S** to save, **Q** to quit")

    with st.spinner("Opening camera..."):
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                status_box.error("❌ Camera not working")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imshow("Dataset Collection (Press S / Q)", frame)
            key = cv2.waitKey(1) & 0xFF

            # Save image
            if key == ord("s"):
                if len(face_locations) == 1:
                    count += 1
                    cv2.imwrite(os.path.join(person_path, f"{count}.jpg"), frame)

                    progress = int((count / max_images) * 100)
                    progress_bar.progress(min(progress, 100))

                    status_box.success(f"✅ Image {count} captured")

                else:
                    status_box.warning("⚠️ Please show only ONE face")

            # Quit
            elif key == ord("q") or count >= max_images:
                break

        cap.release()
        cv2.destroyAllWindows()

    if count > 0:
        status_box.success(f"🎉 Dataset collection completed ({count} images)")
        return True
    else:
        status_box.error("❌ No images captured")
        return False
