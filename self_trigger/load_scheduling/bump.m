function y = bump(x,start_knot,end_knot)
% a bumper function that is 1 at (start_knot,end_knot), 0.5 at start_knot
% and end_knot, 0 elsewhere
    y = 0.5*(sign(x-start_knot)+1)-0.5*(sign(x-end_knot)+1);
end

