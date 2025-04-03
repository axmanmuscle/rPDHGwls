rng(20241025)
num_segments = 5;
len_segments = 400;

gt = zeros([num_segments*len_segments 1]);

ri = randi([-5 5], [num_segments 1]);
C = 0.25*randn([len_segments 1]) + ri(1);
gt(1:len_segments) = ri(1);
for i = 1:num_segments-1
    A = 0.5*randn([len_segments 1]) + ri(i+1);
    C = [C; A];
    gt(i*len_segments+1 : (i+1)*len_segments) = ri(i+1);
end

x = 1:num_segments*len_segments;
% figure; plot(x, C, 'DisplayName', 'Noised'); hold on;
% plot(x, gt, 'DisplayName', 'gt', 'LineWidth', 2)
n = numel(C);

A = @(in) circshift(in, -1) - in;

Amat = zeros(n);
for i = 1:n
    ei = zeros([n 1]);
    ei(i) = 1;
    Amat(:, i) = A(ei);
end

theta = 0.9/norm(Amat)^2;
Bt = chol((1/theta)*eye(n) - Amat*Amat');
B = Bt';
m = size(B, 1);

clear Bt;
B = sparse(B);
Amat = sparse(Amat);

f = @(in) 0.5*norm(in - C, 2)^2;
proxf = @(x, t) proxL2Sq(x, t, C);

proxftilde = @(in, t) [proxf(in(1:n), t); zeros([m 1])];

Rftilde = @(in, t) 2*proxftilde(in, t) - in;


lambda = 10;
tau = 10^-.75

maxIter = 5000;

g = @(in) lambda * norm(in, 1);
ga = @(in) lambda * norm(A(in), 1);
proxgconj = @(in, t) proxConjL1(in, t, lambda);
proxg = @(in, t) in - t * proxgconj(in/t, 1/t);

gtilde = @(in) lambda*g(A(in(1:n)) + B*in(n+1:end));

proxgtilde = @(x, t) x - t*[Amat';B']*proxgconj((theta/t)*(Amat*x(1:n) + B*x(n+1:end)), theta/t);
proxgtildeconj = @(x, t) x - proxgtilde(x, t);
Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;

z0 = zeros(size(C));
tau = taus(tau_idx);
objtilde = @(in) f(proxf(in(1:n), tau)) + gtilde(proxftilde(in, tau));

x0 = zeros([n+m 1]);

[xStar_rpdhg, objVals_rpdhg, alphas] = rPDHG_wls(z0, proxf, proxgconj, ...
    f, g, Amat, B, 'maxIter', maxIter, 'tau0', tau,'beta0', 1, 'verbose', true);
