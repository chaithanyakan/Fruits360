imgFolder=fullfile("fruits360dataset",'fruits-360');
imds=imageDatastore(imgFolder,"LabelSource","foldernames","IncludesubFolders",true);

AppleBraeburn = find(imds.Labels == "Apple Braeburn",1);
figure,imshow(readimage(imds,AppleBraeburn));

tbl=countEachLabel(imds);

minSetCount=min(tbl{:,2});
maxNumImages=100;
minSetCount=min(maxNumImages,minSetCount);

imds=splitEachLabel(imds,minSetCount,"randomize");
countEachLabel(imds);

net=resnet50;
deepNetworkDesigner(net);
figure,plot(net)

title("First section of ResNet50");
set(gca,"YLim",[150 170]);
net.Layers(1);
net.Layers(end);

[Training,Test]=splitEachLabel(imds,0.3,"randomize")

imageSize=net.Layers(1).InputSize

augmentedTrainingSet = augmentedImageDatastore(imageSize, Training);
augmentedTestSet = augmentedImageDatastore(imageSize, Test);

w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
w1 = imresize(w1,5); 
figure
montage(w1)

title('First convolutional layer weights')
featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = Training.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');


predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

testLabels = Test.Labels;


confMat = confusionmat(testLabels, predictedLabels);

confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
mean(diag(confMat))


testImage = readimage(Test,53);
figure,imshow(testImage);



ds = augmentedImageDatastore(imageSize, testImage, 'ColorPreprocessing', 'gray2rgb');


imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');


predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

confusionchart(testLabels, predictedLabels)

