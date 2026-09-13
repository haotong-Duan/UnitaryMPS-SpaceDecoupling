function figure_generate(s)
num_samples = size(s, 2); 
rows = ceil(sqrt(num_samples)); 
cols = ceil(num_samples / rows); 
figure;
set(gcf, 'Color', 'w'); 
tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact'); 
for i = 1:num_samples
    img = reshape(s(:, i), 28, 28)-1;
    nexttile;
    imshow(img, 'InitialMagnification', 'fit', 'Border', 'tight');
    colormap(gray);
    axis off; 
end
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPositionMode', 'auto');
exportgraphics(gcf, 'test.pdf', 'ContentType', 'vector');