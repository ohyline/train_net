import numpy as np
from sklearn.metrics import confusion_matrix
import os

def getConfusionMatrixFromSavedFiles(saved_prediction_path, img_list_path, result_save_path = ''):
	with open(img_list_path, 'rb') as fid:
		lines = fid.readlines()

	predictions = []
	labels = []

	lab_name_map = dict()
	for l in lines:
		tmp = l.split(' ')
		file_name = tmp[0]

		prediction_path = saved_prediction_path + file_name[:-4] + '.npy'
		print 'reading results from: '+ prediction_path
		prediction = np.load(prediction_path)
		label = (int)(tmp[1])
		if label not in lab_name_map :
			name = file_name.split('/')[1]
			lab_name_map[label] = name

		predictions.append(np.argmax(prediction))
		labels.append(label)

	labels = np.array(labels)
	predictions = np.array(predictions)


	header = ','.join(lab_name_map.values())
	unique_labels = lab_name_map.keys()
	cm = confusion_matrix(labels, predictions, unique_labels)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	cm_diag = np.diag(cm)
	avg = np.mean(cm_diag)

	result = np.vstack((cm,cm_diag))
	result = np.hstack((result, np.append(cm_diag, avg)[:,np.newaxis]))

	if result_save_path != '':
		np.savetxt(result_save_path, result, delimiter=',', header=header)

	print(result)

saved_prediction_path = './extract_features/feats/test_merged/prob_final/'
img_list_path = '/mnt/hdd/Chen/place/trainvalsplit_places205/test_places14.txt'
result_path = 'confusion_matrix_merged.csv'

getConfusionMatrixFromSavedFiles(saved_prediction_path, img_list_path, result_save_path = result_path)
