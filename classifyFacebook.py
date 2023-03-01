import trainForFacebook

def buildNewModel(filename):
    newModel = trainForFacebook.trainData(filename)


def checkNewInput(clf_SVM_sig, count_vect, newInput):
    return (clf_SVM_sig.predict(count_vect.transform([newInput])))


def whileLoopClassify(clf_SVM_sig, count_vect):
    print("starting while loop")
    while (1 == 1):
        newDescription = input()
        category = checkNewInput(clf_SVM_sig, count_vect, newDescription)
        print(str(category))


if __name__ == "__main__":
    model = buildNewModel("trainingData.csv")
    clf_SVM_sig = model[0]
    count_vect = model[2]
    whileLoopClassify(clf_SVM_sig, count_vect)
