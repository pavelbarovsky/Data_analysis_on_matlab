clc; clear all; close all;

data = load('wine.data');

% Разделение выборки на обучающую и контрольную выборки
cv = cvpartition(size(data, 1), 'HoldOut', 0.5); 

% Индексы объектов обучающей выборки
trainIdx = training(cv);

% Индексы объектов контрольной выборки
testIdx = test(cv);

% Разделение данных на обучающую и контрольную выборки
trainData = data(trainIdx, :);
testData = data(testIdx, :);

% Разделение признаков и меток классов
x = trainData(:, [2, 11, 13]); % Признаки x, y, z
c = trainData(:, 1); % Метки классов c

x_test = testData(:, [2, 11, 13]); % Признаки x, y, z
c_test = testData(:, 1); % Метки классов c

% Создание и обучение классификатора ближайшего соседа
knnClassifier = fitcknn(x, c, 'NumNeighbors', 1);

% Классификация объектов контрольной выборки
c_pred = predict(knnClassifier, x_test);

% Визуализация объектов в трехмерном пространстве
figure;
scatter3(x(:, 1), x(:, 2), x(:, 3), 'filled', 'MarkerFaceColor', 'b'); % Обучающая выборка (синие точки)
hold on;
scatter3(x_test(:, 1), x_test(:, 2), x_test(:, 3), 'filled', 'MarkerFaceColor', 'g'); % Контрольная выборка (зеленые точки)
xlabel('x (Алкоголь)');
ylabel('y (Оттенок вина)');
zlabel('z (Оптическая плотность)');
legend('Обучающая выборка', 'Контрольная выборка');

% Создание графика с настоящими классами контрольной выборки
figure;
scatter3(x_test(:, 1), x_test(:, 2), x_test(:, 3), [], c_test, 'filled');
xlabel('x (Алкоголь)');
ylabel('y (Оттенок вина)');
zlabel('z (Оптическая плотность)');
title('Контрольная выборка: Настоящие классы');

% Создание графика с классами, которым были отнесены объекты контрольной выборки
figure;
scatter3(x_test(:, 1), x_test(:, 2), x_test(:, 3), [], c_pred, 'filled');
xlabel('x (Алкоголь)');
ylabel('y (Оттенок вина)');
zlabel('z (Оптическая плотность)');
title('Контрольная выборка: Классификация');

% Оценка вероятности ошибочной классификации
misclassificationRate = sum(c_pred ~= c_test) / numel(c_test);
fprintf('Вероятность ошибочной классификации реальных данных: %.2f\n', misclassificationRate);



% Генерация случайных величин для первого класса
n1 = 100; % Количество примеров первого класса
a1 = [2; -2; 0]; % Математическое ожидание для первого класса
R1 = [2 -1 0.1; -1 4 -1; 0.1 -1 2]; % Корреляционная матрица для первого класса
rng(42); % Задаем начальное состояние генератора случайных чисел для воспроизводимости
data1 = mvnrnd(a1, R1, n1); % Генерация случайных величин для первого класса

% Генерация случайных величин для второго класса
n2 = 100;
a2 = [4; 2; -4];
R2 = [2 0.1 -1; 0.1 2 -1; -1 -1 4]; 
data2 = mvnrnd(a2, R2, n2); 

% Визуализация данных
figure;
scatter3(data1(:, 1), data1(:, 2), data1(:, 3), 'filled', 'MarkerFaceColor', 'b'); % Первый класс (синие точки)
hold on;
scatter3(data2(:, 1), data2(:, 2), data2(:, 3), 'filled', 'MarkerFaceColor', 'r'); % Второй класс (красные точки)
xlabel('X');
ylabel('Y');
zlabel('Z');
legend('Первый класс', 'Второй класс');
title('Визуализация данных');

% Создание обучающей и контрольной выборок
trainData = [data1; data2];
trainLabels = [ones(n1, 1); 2*ones(n2, 1)];

nTest = round((n1 + n2) / 2); % Количество объектов в контрольной выборке

testIndices = randperm(n1 + n2, nTest); % Случайный выбор индексов для контрольной выборки
testData = trainData(testIndices, :); % Контрольная выборка
testLabelsTrue = trainLabels(testIndices); % Настоящие классы объектов контрольной выборки

trainData(testIndices, :) = []; % Удаление объектов контрольной выборки из обучающей выборки
trainLabels(testIndices) = []; % Удаление классов объектов контрольной выборки из обучающей выборки

% Создание и обучение классификатора ближайших соседей
knnClassifier = fitcknn(trainData, trainLabels, 'NumNeighbors', 1);

% Классификация объектов контрольной выборки
predictedLabels = predict(knnClassifier, testData);

% Визуализация объектов контрольной выборки с настоящими классами
figure;
scatter3(testData(:, 1), testData(:, 2), testData(:, 3), [], testLabelsTrue, 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Контрольная выборка с настоящими классами');

% Визуализация объектов контрольной выборки с предсказанными классами
figure;
scatter3(testData(:, 1), testData(:, 2), testData(:, 3), [], predictedLabels, 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Контрольная выборка с предсказанными классами');

misclassificationRate = sum(predictedLabels ~= testLabelsTrue) / numel(testLabelsTrue);
fprintf('Вероятность ошибочной классификации смоделированных данных: %.2f\n', misclassificationRate);