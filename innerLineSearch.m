function [xOut, zOut, tau, nsxmx, thetak] = innerLineSearch(xIn, zIn, proxf, proxgconj, beta, taukm1, thetakm1, alpha, A, At, Bt)
% this applyS will do PDHG iterations with Malitsky's line search

%%% line search params
% beta = 0.8;
delta = 0.99;
mu = 0.8;
runLineSearch = true;

xLast1 = xIn - taukm1 * At(zIn);
xLast2 = -taukm1 * Bt(zIn);

xk = proxf( xIn - taukm1 * At( zIn ), taukm1 );

tauk = taukm1 * sqrt(1+thetakm1);

complexNorm = @(x) sqrt( real( dotP( x(:), x(:) ) ) );

if runLineSearch
    accept = false;
    subiter = 1;
    while ~accept
        %fprintf('gpdhg wls subiter %d tau %f\n', subiter, tauk)
        subiter = subiter + 1;
        thetak = tauk / taukm1;
        xbar_k = xk + thetak*(xk - xIn);
        zkp1 = proxgconj(zIn + beta * tauk * A(xbar_k), beta*tauk);
        
        tmpL = At(zkp1) - At(zIn);
        if isreal( tmpL )
          left_term = sqrt(beta) * tauk * norm( tmpL );
        else
          left_term = sqrt(beta) * tauk * complexNorm( tmpL );
        end
    
        tmpR = zkp1 - zIn;
        if isreal( tmpR )
          right_term = delta * norm( zkp1(:) - zIn(:) );
        else
          right_term = delta * complexNorm( zkp1 - zIn );
        end
    
        if left_term <= right_term
            accept = true;
        else
            tauk = tauk * mu;
        end
    end
else
    thetak = tauk / taukm1;
    xbar_k = xk + thetak*(xk - xIn);
    zkp1 = proxgconj(zIn + beta * tauk * A(xbar_k), beta*tauk);
end
xOut = (1-2*alpha)*xIn + 2*alpha*xk;
zOut = (1-2*alpha)*zIn + 2*alpha*zkp1;

xNew1 = xOut - taukm1 * At( zOut );
xNew2 = -taukm1 * Bt( zOut );

sxmx1 = (1/alpha) * ( xNew1 - xLast1 );
sxmx2 = (1/alpha) * ( xNew2 - xLast2 );

if isreal( sxmx1 ) && isreal( sxmx2 )
  nsxmx = sqrt( norm( sxmx1(:),2)^2 + norm( sxmx2(:),2)^2 );
else
  nsxmx = sqrt( complexNorm( sxmx1(:) )^2 + complexNorm( sxmx2(:) )^2 );
end

tau = tauk;

end


