function figure_bas(s, sqrtd, K)
% idx = randperm(size(s, 2), K);
% s = s(:, idx);
num_samples = size(s, 2); 
if K ~=4
    rows = ceil(sqrt(num_samples));
    cols = ceil(num_samples / rows);
end
if K ==4
    rows = 1;            
    cols = num_samples;  
end
figure;
set(gcf, 'Color', 'w');
tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:num_samples
    img = reshape(s(:, i), sqrtd, sqrtd) - 1;
    nexttile;
    imshow(img, 'InitialMagnification', 'fit', 'Border', 'tight');
    colormap(gray);
    axis off;
    hold on;
    rectangle('Position', [0.5, 0.5, sqrtd, sqrtd], ...
              'EdgeColor', 'black', 'LineWidth', 1);
    hold off;
end

set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPositionMode', 'auto');
exportgraphics(gcf, 'test.pdf', 'ContentType', 'vector');
