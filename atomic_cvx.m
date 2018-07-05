function u_recon = atomic_cvx(b, A, epsilon) 
%#ok<*STOUT>
% nobody likes you 
beep off

[~,dim] = size(A);

% solve min_u ||u||_1 st. ||Au - b||_2 <= epsilon 
cvx_begin
    cvx_precision best
    variable u_recon(dim)
    minimise(norm(u_recon,1))
    subject to 
        % find u within epsilon of the true value
        norm(A*u_recon-b,2) <= epsilon
cvx_end
end