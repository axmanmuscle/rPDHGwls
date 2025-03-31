function [xOut, objVals, alphasUsed] = rPDHG_wls( x0, proxf, proxgconj, f, g, A, B, varargin )
  % [xOut, objVals, alphasUsed] = rPDHG_wls( x0, proxf, proxgconj, f, g, A, B, varargin )
  %
  % Implements the relaxed PDHG method with line search. Solves the problem
  %  min  f(x) + g(Ax)
  %   x
  %
  % INPUTS:
  %   x0 - initial guess
  %   proxf - the proximal operator of f. takes two arguments, the input and the scaling.
  %   proxgconj - the proximal operator of g*. takes two arguments, the input and the scaling.
  %   f - handle to the f function
  %   g - handle to the g function
  %   A - the operator A
  %   B - the operator B such that AA^* + BB^* = a*I, a scaled identity
  %
  % OUTPUTS:
  %   xOut - the final answer (lowest objective value calculated from f, g)
  %   objVals - the objective values at each iteration
  %   alphasUsed - which alpha parameter was chosen at each iteration (outer line search)

  p = inputParser;
  p.addParameter( 'maxIter', 100, @ispositive );
  p.addParameter( 'z0', [], @isnumeric );
  p.addParameter( 'beta0', 1, @isnumeric);
  p.addParameter( 'tau0', 1, @isnumeric);
  p.addParameter( 'verbose', false );
  p.parse( varargin{:} );
  maxIter = p.Results.maxIter;
  z0 = p.Results.z0;
  beta0 = p.Results.beta0;
  tau0 = p.Results.tau0;
  verbose = p.Results.verbose;

  if isnumeric(A) && isnumeric(B)
    applyA = @(x) A*x;
    applyAt = @(x) A'*x;
    applyBt = @(x) B'*x;
  
    %theta = diag( A*A' + B*B' );
    %theta = theta(1);

  elseif isa(A, 'function_handle') && isa(B, 'function_handle')
    applyA = @(x) A(x, 'notransp');
    applyAt = @(x) A(x, 'transp');
    applyBt = @(x) B(x, 'transp');

  else
      disp('both A, B must be numeric or functions');
      return
  end
  if numel( z0 ) == 0
      z0 = applyA( x0 );
  end
  
  if nargout > 1
      objVals = zeros( maxIter, 1 );
  end
  
  if nargout > 2
      alphasUsed = zeros( maxIter, 1 );
  end

  
  objFun = @(x) f(x) + g( applyA(x) );
  
  doLineSearch = true;
  doLineSearchTest = true;

  minObj = Inf;
  minObjSol = zeros(size(x0));
  
  %%% parameters
  alpha_bar = 0.5; % alpha_bar
  eps = 0.03; % eps for (1 - eps) || rbar_k || in linesearch                                                             
  epsHat = 0.05; % for line search test
  alpha0 = 20; % starting alpha
  alpha_change = 1/1.4; % factor for change in alpha during linesearch
  
  k = ceil( -log( alpha0 / alpha_bar ) / log( alpha_change ) );
  alphas = alpha0 .* ( alpha_change.^(0:k) );
  alphas(end) = alpha_bar;
  
  S = @(xIn, zIn, tauk, alpha, thetak) innerLineSearch(xIn, zIn, proxf, proxgconj, beta0, tauk, thetak, alpha, applyA, applyAt, applyBt);

  S_nols = @(xIn, zIn, tauk, alpha) innerIters(xIn, zIn, proxf, proxgconj, beta0, tauk, alpha, applyA, applyAt, applyBt);

  nAlphas = numel( alphas );
  normRks = zeros( nAlphas, 1 );
  tauks = cell( 1, nAlphas );
  xks = cell( 1, nAlphas );
  zks = cell( 1, nAlphas );
  thetaks = cell(1, nAlphas);

  tauk = tau0;
  xOut = x0;
  zOut = z0;
  thetak = 1;
  
  normRk = sqrt( real( dotP(x0, x0) ) );
  
  for optIter = 1:maxIter

      lastNormRk = normRk;
  
      if doLineSearch == true
  
          parfor alphaIndx = 1 : nAlphas
              alpha = alphas( alphaIndx );
              % this is eq (6) in Boyd's LS paper
              % performs xkp1 = xk + alphak(Sxk - xk)
              [xOutp1, zOutp1, taukp1, ~, thetaAlpha] = S(xOut, zOut, tauk, alpha, thetak);

              % this now gets the next step, rkalpha = Sxk+1 - xk+1
              % [xOutAlpha, zOutAlpha, taukAlpha, nrkAlpha, thetaAlpha] = S(xOutp1, zOutp1, taukp1, alpha, thetakp1);
              %%% just do PDHG iters here?
              [xOutAlpha, zOutAlpha, nrkAlpha] = S_nols(xOutp1, zOutp1, tauk, alpha);
              tauks{alphaIndx} = taukp1;
              xks{alphaIndx} = xOutAlpha;
              zks{alphaIndx} = zOutAlpha;
              thetaks{alphaIndx} = thetaAlpha;
              
              normRks( alphaIndx ) = nrkAlpha;
          end
  
          normRk_bar = normRks(end);
          bestAlphaIndx = find( normRks <= ( 1-eps ) * normRk_bar, 1 );
          if numel( bestAlphaIndx ) == 0, bestAlphaIndx = numel( normRks ); end
  
          alphaUsed = alphas( bestAlphaIndx );
          tauk = tauks{ bestAlphaIndx };
          xOut = xks{ bestAlphaIndx };
          zOut = zks{ bestAlphaIndx };
          normRk = normRks(bestAlphaIndx);
          thetak = thetaks{bestAlphaIndx};
  
      else
  
          alphaUsed = alpha_bar;
          % xk = xk + alpha_bar * rk;
          [xOut, zOut, tauk, normRk, thetak] = S(xOut, zOut, tauk, alpha_bar, thetak);
      end
  
      objValue = objFun( xOut );

      if objValue < minObj
          minObj = objValue;
          minObjIter = optIter;
          minObjSol = xOut;
      end
  
      if nargout > 1
          objVals( optIter ) = objValue;
      end
      if nargout > 2
          alphasUsed( optIter ) = alphaUsed;
      end
  
      if doLineSearchTest == true
          doLineSearch = false;
          if normRk * lastNormRk > 0
  
            if alphaUsed ~= alpha_bar  ||  normRk / lastNormRk < 1 - epsHat  % My own heuristic
                % The values of S are changing quickly in this region
                doLineSearch = true;
            end
          end
      end
  
      if verbose == true
        fprintf('iter: %d   objVal: %d   res: %d  alpha: %d\n', optIter, objVals(optIter), normRk, alphaUsed );
      end
  
  end

  if objVals(end) > minObj
      fprintf('minimum objective occured on iteration %d\n', minObjIter);
      xOut = minObjSol;
  end

end