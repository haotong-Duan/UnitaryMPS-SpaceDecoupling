function bas_vec = generate_bas(H, W)
    bas = [];
    % ----- Bars -----
    for pattern = 1:(2^W - 2)  
        bits = bitget(pattern, 1:W);
        img = repmat(bits, H, 1);    
        bas = [bas, img(:)];         
    end
    % ----- Stripes -----
    for pattern = 1:(2^H - 2)
        bits = bitget(pattern, 1:H);
        img = repmat(bits', 1, W);   
        bas = [bas, img(:)];
    end
    bas_vec = bas;   
    bas_vec = bas_vec + 1;
end
