function Plot_mean_var(X,Y,E)
%BOUNDEDLINE Plot a line with shaded error/confidence bounds
%X : the vector or matrix for axis X, size(X,2) is the number of line what
%to draw
%Y:  the vector or matrix for axis Y
%E: the corresonding error or variance
% Example:
%
 
if nargin==0
        x = linspace(0, 2*pi, 50);
        y1 = sin(x);
        y2 = cos(x);
        e1 = rand(size(y1))*.5+.5;
        e2 = [.25 .5];

        ax(1) = subplot(2,2,1);
        [l,p] = boundedline(x, y1, e1, '-b*');
        outlinebounds(l,p);
        title('Opaque bounds, with outline');

        ax(2) = subplot(2,2,2);
        boundedline(x, [y1;y2], rand(length(y1),2)*.5+.5, 'alpha');
        title('Transparent bounds');

        ax(3) = subplot(2,2,3);
        boundedline([y1;y2], x, e1(1), 'orientation', 'horiz')
        title('Horizontal bounds');

        ax(4) = subplot(2,2,4);
        boundedline(x, repmat(y1, 4,1), permute(0.5:-0.1:0.2, [3 1 2]), ...
            'cmap', cool(4), 'transparency', 0.5);
        title('Multiple bounds using colormap');
        

      
        
else
  
  
        [l,p]=boundedline(X,Y,E, 'alpha');
        set(l,'LineWidth',1.8)

       
end

end

