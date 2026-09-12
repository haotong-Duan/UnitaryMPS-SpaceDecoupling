function figure_mnist(s, k)
    num_samples = size(s, 2); 
    rows = ceil(sqrt(num_samples)); 
    cols = ceil(num_samples / rows); 
    figure;
    set(gcf, 'Color', 'w'); 
    tiledlayout(rows, cols, 'Padding', 'compact', 'TileSpacing', 'compact'); 
    for i = 1:num_samples
        img = reshape(s(:, i), 28, 28);
        rgb_img = ones(28, 28, 3); 
        ones_indices = find(img == 1);
        idx = find(ones_indices> k, 1);
        blue_indices = ones_indices(1:idx-1);
        black_indices = ones_indices(idx:end);
            for idx = blue_indices'
                [row, col] = ind2sub([28, 28], idx);
                rgb_img(row, col, :) = [0, 0, 1]; 
            end
            for idx = black_indices'
                [row, col] = ind2sub([28, 28], idx);
                rgb_img(row, col, :) = [0, 0, 0]; 
            end
        nexttile;
        imshow(rgb_img, 'InitialMagnification', 'fit', 'Border', 'tight');
        axis off; 
    end
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPositionMode', 'auto');
exportgraphics(gcf, 'figure_generate_given_q.pdf', 'ContentType', 'vector');
end    
