clc; clear all; close all;

data = load('wine.data');
n = 100; % количество объектов
x = data(1:n, 2); 
y = data(1:n, 11);
c = data(1:n, 1);

k = max(c); % количество кластеров
features = [x, y];
[idx, centers] = kmeans(features, k);

% по кластерам
figure; hold on;
colors = {'b', 'm', 'y', 'r', 'c', 'g', 'k'}; % цвета для каждого кластера
for i = 1:k
    cluster_points = features(idx == i, :);
    scatter(cluster_points(:, 1), cluster_points(:, 2), 25, colors{i});
end
scatter(centers(:, 1), centers(:, 2), 100, 'k', 'filled'); % отображение центров кластеров
legend('Кластер 1', 'Кластер 2', 'Центры кластеров');
xlabel('Содержание алкоголя');ylabel('Цвет вина');
title('Разбиение объектов наблюдения по кластерам');

% по классам
figure; hold on;
for i = 1:k
    class_points = features(c == i, :);
    scatter(class_points(:, 1), class_points(:, 2), 25, colors{i});
end
scatter(centers(:, 1), centers(:, 2), 100, 'k', 'filled'); % отображение центров кластеров
legend('Класс 1', 'Класс 2', 'Центры кластеров');
xlabel('Содержание алкоголя'); ylabel('Цвет вина');
title('Разбиение объектов наблюдения по классам');

% Вычисление суммы квадратов расстояний от каждого объекта наблюдения до центра соответствующего кластера
sum_of_squares = 0;
for i = 1:n
    cluster_center = centers(idx(i), :);
    sum_of_squares = sum_of_squares + sum((features(i, :) - cluster_center).^2);
end
disp(['Сумма квадратов расстояний: ', num2str(sum_of_squares)]);

% Моделирование данных
n1 = 1000; a1 = [1; -1]; R1 = [1 -0.1; -0.1 2];
n2 = 100; a2 = [4; 2]; R2 = [2 -0.1; -0.1 1];
rng(1);
X1 = mvnrnd(a1, R1, n1); Y1 = mvnrnd(a2, R2, n2);

X = [X1; Y1];
k2 = 2; % Количество кластеров
[idx2, centers2] = kmeans(X, k2);

% по кластерам
figure; hold on;
scatter(X(:, 1), X(:, 2), [], idx2);
scatter(centers2(:, 1), centers2(:, 2), 100, 'k', 'filled');
title('Разбиение по кластерам');
legend('Кластер 1 и 2', 'Центры кластеров');
xlabel('X'); ylabel('Y');

distances = zeros(size(X, 1), 1);
for i = 1:size(X, 1)
    distances(i) = sum((X(i, :) - centers2(idx2(i), :)).^2);
end
sum_of_distances = sum(distances);
disp(['Сумма квадратов расстояний: ' num2str(sum_of_distances)]);

