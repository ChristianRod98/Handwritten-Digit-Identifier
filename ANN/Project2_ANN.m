% Christian Rodriguez
% CSCI4840 Machine Learning
% Project 2
% This Script creates an artificial neural network for the recognition of
% handwritten digits.

clc;
clear;
close all;

% Initializing the inputs and outputs that will be used for the ANN.
% loadMNIST files provide functions to reshape the provided input and
% output files to be plased in rows and colomns of images 784 x 60,000 and labels
% 60,000 x 1.
imageFiles =loadMNISTImages('training_set');
labelFiles = loadMNISTLabels('training_label');

% Reassigns the values equal to 10 from the labelFiles to 0.
size(labelFiles)
labelFiles(labelFiles==0)=10;
labelFiles = dummyvar(labelFiles);
size(labelFiles)

% Determins the number of hidden layers what will be plased in the neural
% network and the training method for training the network.
hiddenLayerSize = [200];
trainFcn = 'traingd';

% Displays the number of subplots, the handwritten images, with their
% guessed numeral value that the neural network assigns in the title.
figure
colormap('gray');
for i =1:36
    subplot(6,6,i)
    value = reshape(imageFiles(:,i), [28,28]);
    imagesc(value);
    
    value_text = find(labelFiles(i,:)==1);
    
    title(['Value: ',num2str(value_text) ]);
end
pause

% Initiates the input and output for the neural network.
x = imageFiles;
t = labelFiles;
% Initializes the testing for the artificial neural network.
imageFiles =loadMNISTImages('test_set');
labelFiles = loadMNISTLabels('test_label');
hits = 0;
misses = 0;

% Creates the artificial neural network and assigns the number of hidden
% layers and training method.
net = patternnet(hiddenLayerSize, trainFcn, imageFiles, labelFiles, test_ImageFiles, test_LabelFiles);
% Initializes parameters for adjusting the neural network.
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Default for ANN is mse.
net.performFcn = 'mse';
[net,tr] =train(net,x,t);

% Creates a loop for testing the outputs and keeps track of the correct and
% incorrect guesses.
for i = 1: size(test_ImageFiles)
    M = test_ImageFiles(:, i);
    test_imageFiles = reshape(M,28,28);
    guess = net(M);
    
    [val,idx] = max(guess);
    [val2,idx2] = max(test_LabelFiles(i, :));
        if idx == idx2
            hits = hits + 1;
        else
            misses = misses +1;
        end
end

figure
disp(hits);
disp(misses);

acc = hits/(hits + misses);
figure
disp(acc);
% Reshapes the images back to its  original state to display it in the
% figure loop.
dispimg = test_ImageFiles(:,1);
test_ImageFiles = reshape(dispimg, 28, 28);
disguess = net(dispimg);
% Displays the newly reshaped image files.
figure
imshow(imageFiles);

disp(disguess);
disp(acc);
view(net)
