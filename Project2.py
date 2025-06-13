import pandas as pd
import numpy as np


def greedySearchForwardSelection(numFeatures):
    df = pd.read_csv(file_name,sep=r'\s+',header=None) # setup correct df
    print(f"This dataset has {df.shape[1]-1} features (not including the class attribute), with {df.shape[0]} instances.")
    candidateFeatures = pd.DataFrame(columns=['Accuracy', 'Feature']) # load df for all feature's individual accuracies
    NearestNeighbor = NNClassifier()
    LOOV_validation = LOOV()


    # load in all individual accuracies and corresponding feature
    for i in range(numFeatures):
        featureEvaluation = LOOV_validation.validate([i+1], NearestNeighbor, df);
        candidateFeatures.loc[i, 'Accuracy'] = featureEvaluation;
        candidateFeatures.loc[i, 'Feature'] = i+1;
    
    # empty set for selected features, use to keep track of optimal feature combination
    selectedFeatures = pd.DataFrame(columns=['Accuracy', 'Feature'])
    # keeps track of current best overall feature combination accuracy
    bestSelectedFeatures = []
    labels = df.iloc[:,0]
    two_counts = 0
    one_counts = 0
    # count most common class
    for i in range(len(labels)):
        if (labels.iloc[i] == 1):
            one_counts += 1
        if (labels.iloc[i] == 2):
            two_counts += 1    
    if (two_counts > one_counts):
        defaultRate = two_counts/len(df)
    else:
        defaultRate = one_counts/len(df)
    print(f"Please wait while I normalize the data... Done!")
    print(f"Running nearest neighbor with no features (default rate), using “leaving-one-out” evaluation, I get an accuracy of {defaultRate * 100:.1f}%")

    currAccuracy = 0
    bestAccuracy = defaultRate
    bestAccuracyThisIteration = defaultRate # tracker for local optima
    print(f"Beginning search.")
    while len(selectedFeatures) < numFeatures:
        bestFeature = -1
        localBestAccuracy = -1
        localBestFeatures = []
        for i in range(len(candidateFeatures)):
            consideredFeatures = selectedFeatures['Feature'].tolist()  # Hold best combination from previous iteration
            consideredFeatures.append(candidateFeatures.loc[i, 'Feature'])  # Append the current feature being tested with best combination from previous iteration
            currAccuracy = LOOV_validation.validate(consideredFeatures, NearestNeighbor, df)
            # print features currently being tested
            print(f"    Using feature(s) {{{', '.join(map(str, consideredFeatures))}}} accuracy is {currAccuracy * 100:.1f}%")
            if localBestAccuracy < currAccuracy:
                localBestAccuracy = currAccuracy # Select locally best accuracy
                localBestFeatures = consideredFeatures
                bestFeature = i 

            # Keep track of global best features
            if bestAccuracy < localBestAccuracy:
                bestAccuracy = localBestAccuracy
                bestSelectedFeatures = localBestFeatures

       
        if bestFeature != -1:
            selectedFeatures = pd.concat([selectedFeatures, candidateFeatures.iloc[[bestFeature]].reset_index(drop=True)], ignore_index=True)
            candidateFeatures = candidateFeatures.drop(bestFeature).reset_index(drop=True)
            oldBest = bestAccuracyThisIteration # hold previous best for comparison to indicate accuracy drop
            bestAccuracyThisIteration = LOOV_validation.validate(localBestFeatures, NearestNeighbor, df)
            if (oldBest > bestAccuracyThisIteration):
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print(f"Feature set {{{', '.join(map(str, localBestFeatures))}}} was best, accuracy is {(bestAccuracyThisIteration)* 100:.1f}%")

                
        else:
            break # Stop when no more features left

    print(f"Finished search!! The best feature subset is {{{', '.join(map(str, bestSelectedFeatures))}}}, which has an accuracy of {bestAccuracy* 100:.1f}%")
    
def greedySearchBackwardSelection(numFeatures):
    df = pd.read_csv(file_name,sep=r'\s+',header=None) # setup correct df
    print(f"This dataset has {df.shape[1]-1} features (not including the class attribute), with {df.shape[0]} instances.")
    candidateFeatures = pd.DataFrame(columns=['Accuracy', 'Feature']) # load df for all feature's individual accuracies
    NearestNeighbor = NNClassifier()
    LOOV_validation = LOOV()

    # load in all individual accuracies and corresponding feature
    for i in range(numFeatures):
        featureEvaluation = LOOV_validation.validate([i+1], NearestNeighbor, df);
        candidateFeatures.loc[i, 'Accuracy'] = featureEvaluation;
        candidateFeatures.loc[i, 'Feature'] = i+1;
    
    selectedFeatures = pd.DataFrame(columns=['Accuracy', 'Feature'])
    bestSelectedFeatures = []
    labels = df.iloc[:,0]
    two_counts = 0
    one_counts = 0
    for i in range(len(labels)):
        if (labels.iloc[i] == 1):
            one_counts += 1
        if (labels.iloc[i] == 2):
            two_counts += 1    
    if (two_counts > one_counts):
        defaultRate = two_counts/len(df)
    else:
        defaultRate = one_counts/len(df)

    print(f"Please wait while I normalize the data... Done!")
    print(f"Running nearest neighbor with no features (default rate), using “leaving-one-out” evaluation, I get an accuracy of {defaultRate * 100:.1f}%")
    currAccuracy = 0
    bestAccuracy = defaultRate
    bestAccuracyThisIteration = defaultRate
    print(f"Beginning search.")
    while len(selectedFeatures) < numFeatures:
        bestFeature = -1
        localBestAccuracy = -1

        for i in range(len(candidateFeatures)):
            consideredFeatures = candidateFeatures['Feature'].tolist()  # Hold best combination from previous iteration
            consideredFeatures.remove(candidateFeatures.loc[i, 'Feature'])  # Append the current feature being tested with best combination from previous iteration
            currAccuracy = currAccuracy = LOOV_validation.validate(consideredFeatures, NearestNeighbor, df)
            # print features currently being tested
            print(f"    Using feature(s) {{{', '.join(map(str, consideredFeatures))}}} accuracy is {currAccuracy * 100:.1f}%")

            if localBestAccuracy < currAccuracy:
                localBestAccuracy = currAccuracy # Select locally best accuracy
                localBestFeatures = consideredFeatures
                bestFeature = i 

            if bestAccuracy < localBestAccuracy:
                bestAccuracy = localBestAccuracy
                bestSelectedFeatures = localBestFeatures
                

       
        if bestFeature != -1:
            selectedFeatures = pd.concat([selectedFeatures, candidateFeatures.iloc[[bestFeature]].reset_index(drop=True)], ignore_index=True)
            candidateFeatures = candidateFeatures.drop(bestFeature).reset_index(drop=True)
            oldBest = bestAccuracyThisIteration # hold previous best for comparison to indicate accuracy drop
            bestAccuracyThisIteration = LOOV_validation.validate(localBestFeatures, NearestNeighbor, df)
            if (oldBest > bestAccuracyThisIteration):
                print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print(f"Feature set {{{', '.join(map(str, localBestFeatures))}}} was best, accuracy is {bestAccuracyThisIteration* 100:.1f}%")

        else:
            break # Stop when no more features left
    print(f"Finished search!! The best feature subset is {{{', '.join(map(str, bestSelectedFeatures))}}}, which has an accuracy of {bestAccuracy* 100:.1f}%")


class NNClassifier():
    def __init__(self):
        self.X_norm = None
        self.means = None
        self.std = None
        self.train_x = None
        self.train_y = None

    def train(self, X):
        self.train_x = X.iloc[:,1:]
        self.means = self.train_x.mean()
        self.std = self.train_x.std()
        self.X_norm = (self.train_x - self.means)/self.std
        self.train_y = X.iloc[:,0] # store labels

    def test(self, X):
        # Normalized instance
        train_x = X.iloc[1:]
        X_norm  = (train_x - self.means)/self.std
        X_norm = X_norm.values.reshape(1,-1)
        distances = np.sqrt(np.sum((self.X_norm-X_norm)**2, axis = 1))     #Euclidean distance calculation axis needs to be 1 for correct element operations

        predicted_label_index = np.argmin(distances)                #Getting the index of the nearest neighbor for euclidean distance

        predicted_val = self.train_y.iloc[predicted_label_index]        #Getting Predicted value from the nn index
        return predicted_val

class LOOV():
    #leave one instance out, train_x is set to the rest
    def validate(self, feature, classifier, dataset):
        # count correct predictions
        correct = 0
        # include labels in selected feature for classifier to store
        selectedFeatures = [0] + feature
        datasetWithOnlyLabelAndSelectedFeature = dataset.iloc[:, selectedFeatures]
        y_actual = dataset.iloc[:,0]
        for i in range(len(dataset)):
            leaveOutIndex = i
            classifier.train(datasetWithOnlyLabelAndSelectedFeature.drop(leaveOutIndex, axis = 0)) # leave out row i
            y_pred = classifier.test(datasetWithOnlyLabelAndSelectedFeature.iloc[leaveOutIndex])
            if (y_pred == y_actual.iloc[leaveOutIndex]):
                correct += 1
        Accuracy = correct/len(dataset)
        return Accuracy
            
print(f"Welcome to Thien Pham Feature Selection Algorithm.")
print(f"Type in the name of the file to test : ")
file_name = input()
print(f"Type the number of the algorithm you want to run.\n")
print(f"    Forward Selection")
print(f"    Backward Selection")
algorithm_type = int(input())
df = pd.read_csv(file_name,sep=r'\s+',header=None)
if (algorithm_type == 1):
    greedySearchForwardSelection(df.shape[1]-1)
if (algorithm_type == 2):
    greedySearchBackwardSelection(df.shape[1]-1)
