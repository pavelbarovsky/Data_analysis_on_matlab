clc; clear all; close all;
% Заданные параметры
n = 100;  % Количество точек
a = 10;   % Параметр a
b = -0.1; % Параметр b
q = 1;    % Дисперсия

% Загрузка данных из файла
data = load('wine.data');
x = data(1:n, 2); % Значения признака x из столбца 2
y = data(1:n, 11); % Значения признака y из столбца 11

% Генерация гауссовского белого шума
rng('default'); % Для воспроизводимости результатов
epsilon = sqrt(q) * randn(n, 1); % Генерация шума

% Построение модели линейной регрессии с учетом шума
X = [x ones(n, 1)]; % Матрица признаков X (столбец x и столбец из единиц)
y_with_noise = a * x + b + epsilon;

% Оценка параметров a и b с использованием метода наименьших квадратов
coefficients = (X' * X) \ (X' * y_with_noise);
a_estimated = coefficients(1);
b_estimated = coefficients(2);
disp(['Оценка параметра a: ' num2str(a_estimated)]);
disp(['Оценка параметра b: ' num2str(b_estimated)]);

% Предсказание значений y на основе оцененных параметров
y_predicted = a_estimated * x + b_estimated;

% Вычисление суммы квадратов отклонений
SStot = sum((y_with_noise - mean(y_with_noise)).^2);
SSres = sum((y_with_noise - y_predicted).^2);

% Вычисление коэффициента детерминации (R2)
R2 = 1 - SSres / SStot;
disp(['Коэффициент детерминации (R2): ' num2str(R2)]);

% Визуализация точек и прямой
figure;
plot(x, y_with_noise, 'o')
hold on;
plot(x, y_predicted, 'r', 'LineWidth', 2); % Прямая y=ax+b
hold off;

xlabel('x');
ylabel('y');
title('Линейная регрессия реальных данных');
legend('Точки (x, y) с шумом', 'Прямая y=ax+b');


% ______________________________________
xi = linspace(0, 1, n)';

% Моделирование точек yi = axi + b + ε
yi = a * xi + b + epsilon;

% Построение изображения точек и линии регрессии
figure;
plot(xi, yi, 'o'); % Точки (xi, yi)
hold on;
x_fit = linspace(0, 1, n); % Значения x для построения линии регрессии
y_fit = a * x_fit + b; % Значения y для линейной регрессии
plot(x_fit, y_fit, 'r', 'LineWidth', 2); % Линия регрессии
hold off;

xlabel('x');
ylabel('y');
title('Линейная регрессия смоделированных данных');
legend('Точки (x, y)', 'Линия регрессии');

disp('Смоделированные данные:')
% Модель линейной регрессии в явном виде с определенными коэффициентами
disp('Модель линейной регрессии:');
disp(['y = ' num2str(a) 'x + ' num2str(b)]);

% Вычисление коэффициента детерминации
y_mean = mean(yi); % Среднее значение y
SStot = sum((yi - y_mean).^2); % Общая сумма квадратов отклонений
SSres = sum((yi - (a * xi + b)).^2); % Сумма квадратов остатков
R2 = 1 - SSres / SStot; % Коэффициент детерминации

disp(['Коэффициент детерминации (R2): ' num2str(R2)]);

