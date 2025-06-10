import cv2 as cv
import numpy as np
import tensorflow as tf



class SkinDetect:
    def loading_model(self):
        """
        This funtion loads the model saved from keras.
        """

        skin_cancer_model = tf.keras.models.load_model("trained_model/SKIN_CANCER_DETECTION")

        return skin_cancer_model
    
    def loading_labels(self):
        """
        This function contains labels for types
        """

        labels = {0:'benign',1:'malignant'}

        return labels
        

    def processing_img(self,img_path):
        """
        This function takes image and preprocess it according to model requirements.
        """

        img = cv.imread(img_path,0)
        img = cv.resize(img,(96,96))
        
        # reshaping image for model
        img = img.reshape((1,96,96,1)).astype('float32')
        
        # converting to range between 0-1
        processed_img = img/255
        
        # result = model.predict(img)
        
        # perc = np.amax(result)
        # pred = np.argmax(result[0])
        
        return processed_img
    
    def invertCode(self, result_arr):
        """
        This function converts the one hot encoded data to original form.
        """
        labels = self.loading_labels()
        cancer_type = labels[np.argmax(result_arr)]
        return cancer_type
    
    def DetectionConfidence(self, result_arr):
        confidence = np.amax(result_arr)
        confidence = confidence*100
        return confidence
    
    def detection(self,img_path):

        """
        This function calls model function and detects type of cancer.
        """

        processed_img = self.processing_img(img_path)
        model = self.loading_model()
        detected_type_arr = model.predict(processed_img)
        label_answer = self.invertCode(detected_type_arr)
        confidence = self.DetectionConfidence(detected_type_arr)

        return label_answer, confidence
    


if __name__ == '__main__':
    """
    Main function to check the working of model.
    """

    sd = SkinDetect()

    ans = sd.detection("C:\\Users\\Nitin\\Documents\\projects\\vaishnavi\\skin cancer detection\\demo imgs\\test imgs\\malignant\\melanoma_10117.jpg")
    print(ans)