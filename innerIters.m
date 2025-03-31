function [xOut, zOut, nsxmx] = innerIters(xIn, zIn, proxf, proxgconj, beta, tau, alpha, A, At, Bt)
% this applyS will do PDHG iterations without Malitsky's line search

complexNorm = @(x) sqrt( real( dotP( x(:), x(:) ) ) );
xLast1 = xIn - tau * At(zIn);
xLast2 = -tau * Bt(zIn);

xbar_k = proxf( xIn - tau * At( zIn ), tau );
zbar_k = proxgconj(zIn + beta * tau * A(2*xbar_k - xIn), beta*tau);        
        
xOut = (1-2*alpha)*xIn + 2*alpha*xbar_k;
zOut = (1-2*alpha)*zIn + 2*alpha*zbar_k;

xNew1 = xOut - tau * At( zOut );
xNew2 = -tau * Bt( zOut );

sxmx1 = (1/alpha) * ( xNew1 - xLast1 );
sxmx2 = (1/alpha) * ( xNew2 - xLast2 );

if isreal( sxmx1 ) && isreal( sxmx2 )
  nsxmx = sqrt( norm( sxmx1(:),2)^2 + norm( sxmx2(:),2)^2 );
else
  nsxmx = sqrt( complexNorm( sxmx1(:) )^2 + complexNorm( sxmx2(:) )^2 );
end

end


