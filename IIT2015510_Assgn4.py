import tensorflow as tf
import numpy as np
import cv2
import scipy.spatial.distance as sp

def main():

    """TRAINING"""

    path = 'Faces/'

    # all_weights = []
    # all_mean = []
    face_data = []

    for x in range (5):
        # face_data = []
        for j in range (11):
            filename = str(path + str(x) +'/' + str(j) + '.Jpg')
            temp = np.empty([640,480])
            temp = cv2.imread(filename,0)

            f = temp.flatten()

            face_data.append(f)

    train_faces = np.asarray(face_data)
    mean = np.mean(train_faces, axis = 0)

    for i in range (train_faces.shape[1]):
        for j in range(train_faces.shape[0]):
            train_faces[j][i] = train_faces[j][i] - mean[i]

    cov_mat = np.matmul(train_faces,np.transpose(train_faces))

    eigval, eigvec = np.linalg.eig(cov_mat)

    imp_eig_vec = eigvec[::,0:1]

    eig_faces = np.matmul(np.transpose(imp_eig_vec),train_faces)

    weights = np.matmul(eig_faces,np.transpose(train_faces))

    w = np.reshape(weights,(55))
    st_dev = np.std(w, axis = 0)
    var = st_dev*st_dev
    var = np.reshape(var,[1,1])
    # print(weights)

    # print(eig_faces.shape, weights.shape)

    """TESTING"""

    path2 = 'Test_faces/'
    labels = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
    total_images = 15
    correct = 0

    for k in range (1,16):
        # print('k=',k)
        min_dist = (10)**21
        filename2 = path2 + str(k) + ".Jpg"
        test_im = np.empty([640,480])
        test_im = cv2.imread(filename2,0)

        test = test_im.flatten()
        test_obj = np.reshape(test,[307200,1])

        for i in range (307200):
            test_obj[i][0] = test_obj[i][0] - mean[i]

        # print(test_obj.shape)

        test_face = np.matmul(eig_faces,test_obj)
        test_face = test_face[0][0]

        # print(test_face)

        for i in range (weights.shape[1]):
            
            dist = abs(weights[0][i] - test_face)
            # dist = sp.mahalanobis(weights[0][i],test_face,np.linalg.inv(var))
            
            if ( dist < min_dist ):
                min_dist = dist
                min_i = i
            # print(dist,min,i)

        if ( labels[k-1] == min_i//11 ):
            correct = correct + 1

        print ("Actual class : " , labels[k-1], " Predicted class : ", min_i//11)

    print ( correct, " correct predictions out of ", total_images, " images")
    print ( "Accuracy = ", correct/total_images )

if __name__ == '__main__':
    main()

