function save_images_emnist(K)
    % First you should download the dataset at 'https://www.nist.gov/itl/products-and-services/emnist-dataset'
    data = load('emnist-letters.mat');
    images = data.dataset.train.images;  % original Nx784 
    N = size(images, 1);
    if N < K
        error('Too more');
    end
    % for example: 150 × 784 -> K = 150
    train_images = images(1:K, :);  
    for i = 1:K
        img = reshape(train_images(i, :), 28, 28)';
        train_images(i, :) = img(:)';
    end
    train_x_binary = train_images > 128;
    train_x_binary = train_x_binary' + 1;
    save('emnist_letters_150.mat', 'train_x_binary');
    fprintf('Saved 150 EMNIST Letters images to emnist_letters_150.mat\n');
end
