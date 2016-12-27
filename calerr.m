function [errateTrain, errateTest, predTrain, predTest] = calerr(nn_params, input_layer_size, hidden_layer_size, num_labels, ...
                                                                    TrainX, Trainy, TestX, Testy)


    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    predTrain = predict(Theta1, Theta2, TrainX);
    errateTrain = (1 - mean(double(predTrain == Trainy))) * 100;
    %fprintf('\nTraining Set Error Rate: %f\n', errateTrain);

    predTest = predict(Theta1, Theta2, TestX);
    errateTest = (1 - mean(double(predTest == Testy))) * 100;

    %fprintf('\nTest Set Error Rate: %f\n', errateTest);
end
