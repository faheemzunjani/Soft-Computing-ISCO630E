import tensorflow as tf
import copy
import numpy as np
import cv2
import scipy.spatial.distance as sp

def main():

    """TRAINING"""

    """Principal Component Analysis"""

    path = 'Faces/'

    face_data = []

    for x in range (5):
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

    imp_eig_vec = eigvec[::,0:2]

    eig_faces = np.matmul(np.transpose(imp_eig_vec),train_faces)

    weights = np.matmul(eig_faces,np.transpose(train_faces))

    # print(weights,weights.shape)
    
    """Linear Discriminant Analysis"""

    final_eig_faces = weights

    class_means = np.empty([5,2])
    all_mean = np.mean(final_eig_faces, axis = 1)

    for j in range(5):
        for i in range (int(j*final_eig_faces.shape[1]/5), int((j+1)*final_eig_faces.shape[1]/5)):
            class_means[j][0] = class_means[j][0] + final_eig_faces[0][i]
            class_means[j][1] = class_means[j][1] + final_eig_faces[1][i]

        class_means[j][0] = class_means[j][0]/11
        class_means[j][1] = class_means[j][1]/11

    """Calculating within class scatter s_w and between class scatter s_b"""

    facesw = copy.deepcopy(final_eig_faces)
    facesb = copy.deepcopy(final_eig_faces)

    for i in range ( facesb.shape[1] ):
        facesb[0][i] = facesb[0][i] - all_mean[0]
        facesb[1][i] = facesb[1][i] - all_mean[1]

    for i in range ( facesw.shape[1] ):
        a = int(i/11)
        facesw[0][i] = facesw[0][i] - class_means[a][0]
        facesw[1][i] = facesw[1][i] - class_means[a][1]

    s_b = np.matmul(facesb,np.transpose(facesb))
    s_w = np.matmul(facesw,np.transpose(facesw))

    """Calculation of eigen values and eigen vectors"""

    eigval2, eigvec2 = np.linalg.eig(np.matmul(np.linalg.inv(s_w),s_b))

    imp_eig_vec2 = eigvec2[::,-1:]

    fisher_faces = np.matmul(np.transpose(imp_eig_vec2),final_eig_faces)

    # print(fisher_faces,fisher_faces.shape)

    w1 = np.reshape(fisher_faces,(55))
    st_dev = np.std(w1, axis = 0)
    var = st_dev*st_dev
    var = np.reshape(var,[1,1])

    """TESTING"""

    path2 = 'Test_faces/'
    labels = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
    total_images = 15
    correct = 0

    for k in range (1,16):

        min_dist = (10)**21
        filename2 = path2 + str(k) + ".Jpg"
        test_im = np.empty([640,480])
        test_im = cv2.imread(filename2,0)

        test = test_im.flatten()
        test_obj = np.reshape(test,[307200,1])

        for i in range (307200):
            test_obj[i][0] = test_obj[i][0] - mean[i]

        test_face = np.matmul(eig_faces,test_obj)
        test_face_final = np.matmul(np.transpose(imp_eig_vec2),test_face)
        test_face_final = test_face_final[0][0]

        for i in range (fisher_faces.shape[1]):
            # dist = abs(fisher_faces[0][i] - test_face_final)
            dist = sp.mahalanobis(fisher_faces[0][i],test_face_final,np.linalg.inv(var))
            
            if ( dist < min_dist ):
                min_dist = dist
                min_i = i
            # print(dist,i) 

        if ( labels[k-1] == min_i//11 ):
            correct = correct + 1

        print ("Actual class : " , labels[k-1], " Predicted class : ", min_i//11)

    print ( correct, " correct predictions out of ", total_images, " images")
    print ( "Accuracy = ", correct/total_images )


if __name__ == '__main__':
    main()

