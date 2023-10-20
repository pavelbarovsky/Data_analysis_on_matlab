clc; clear all; close all;

n = 100;
data = csvread('wine.data'); % Загрузка данных из файла wine.data
x = data(1:n, 2); % Извлечение 2-го столбца
y = data(1:n, 11); % Извлечение 11-го столбца

% 1
figure, hold on
for i=1:length(x)
    plot(x(i),y(i),'o')
    xlabel("Кол-во алкоголя")
    ylabel("Оттенок вина")
end

correlation_coefficient = corrcoef(x, y); % Оценка коэффициента корреляции Пирсона
pearson_coef = correlation_coefficient(1, 2); % Извлечение коэффициента корреляции Пирсона из результата
fprintf('Коэффициент корреляции: %.2f\n', pearson_coef);

alpha = 0.05; % Уровень значимости

% Вычисляем корреляцию и получаем необходимые значения
[R, P, ~, ~] = corrcoef(x, y);

if P(1, 2) < alpha
    disp('Отвергаем гипотезу о некоррелированности x и y');
else
    disp('Не получаем достаточных доказательств для отвержения гипотезы о некоррелированности x и y');
end

a = [1, -1];
R = [1  -0.1;
    -0.1 2];

dat = mvnrnd(a, R, n*10);
xo = dat(:,1);
yo = dat(:,2);

figure, hold on
for i=1:length(x)
    plot(xo(i),yo(i),'o')
end
xlabel("Кол-во алкоголя")
ylabel("Оттенок вина")

correlation_coefficient_model = corrcoef(xo, yo); % Оценка коэффициента корреляции Пирсона
pearson_coef_model = correlation_coefficient_model(1, 2); % Извлечение коэффициента корреляции Пирсона из результата
fprintf('Коэффициент корреляции: %.2f\n', pearson_coef_model);

alpha_model = 0.05; % Уровень значимости

% Вычисляем корреляцию и получаем необходимые значения
[R_model, P_model, ~, ~] = corrcoef(xo, yo);

if P_model(1, 2) < alpha_model
    disp('Отвергаем гипотезу о некоррелированности x и y');
else
    disp('Не получаем достаточных доказательств для отвержения гипотезы о некоррелированности x и y');
end
