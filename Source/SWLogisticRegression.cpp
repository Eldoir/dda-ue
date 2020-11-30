#include "SWLogisticRegression.h"

#include "SWDataLR.h"
#include "SWModelLR.h"

USWModelLR * SWLogisticRegression::ComputeModel( USWDataLR * datas )
{
    auto model = NewObject< USWModelLR >();

    try
    {
        const auto maxIterations = 25;
        const auto epsilon = 0.01f;      // stop if all new beta values change less than epsilon (algorithm has converged?)
        const auto jumpFactor = 1000.0f; // stop if any new beta jumps too much (algorithm spinning out of control?)

        model->Betas = ComputeBestBeta( datas->IndepVar, datas->DepVar, maxIterations, epsilon, jumpFactor ); // computing the beta parameters is synonymous with 'training'
    }
    catch ( std::exception e )
    {
        GEngine->AddOnScreenDebugMessage( -1, 1000.f, FColor::Red, FString::Printf( TEXT( "Fatal in ComputeBestBeta: %hs" ), e.what() ) );
    }

    return model;
}

float SWLogisticRegression::TestModel( USWModelLR * model, USWDataLR * testData )
{
    float acc = 0;
    try
    {
        acc = PredictiveAccuracy( testData->IndepVar, testData->DepVar, model->Betas ) / 100.0; // percent of data cases correctly predicted in the test data set.
    }
    catch ( std::exception e )
    {
        GEngine->AddOnScreenDebugMessage( -1, 1000.f, FColor::Red, FString::Printf( TEXT( "Fatal in TestModel: %hs" ), e.what() ) );
    }

    return acc;
}

float SWLogisticRegression::PredictiveAccuracy( TArray< TArray< float > > & xMatrix, TArray< float > & yVector, TArray< float > & bVector )
{
    // returns the percent (as 0.00 to 100.00) accuracy of the bVector measured by how many lines of data are correctly predicted.
    // note: this is not the same as accuracy as measured by sum of squared deviations between
    // the probabilities produceed by bVector and 0.0 and 1.0 data in yVector
    // For predictions we simply see if the p produced by b are >= 0.50 or not.

    if ( xMatrix.Num() == 0 || yVector.Num() == 0 || bVector.Num() == 0 )
        return 0;

    const auto xRows = xMatrix.Num();
    const auto xCols = xMatrix[ 0 ].Num();
    const auto yRows = yVector.Num();
    const auto bRows = bVector.Num();
    if ( xCols != bRows || xRows != yRows )
        throw new std::exception( "Bad dimensions for xMatrix or yVector or bVector in PredictiveAccuracy()" );

    auto numberCasesCorrect = 0;
    auto numberCasesWrong = 0;
    auto pVector = ConstructProbVector( xMatrix, bVector ); // helper also used by LogisticRegressionNewtonParameters()
    const auto pRows = pVector.Num();
    if ( pRows != xRows )
        throw new std::exception( "Unequal rows in prob vector and design matrix in PredictiveAccuracy()" );

    for ( auto i = 0; i < yRows; ++i ) // each dependent variable
    {
        if ( pVector[ i ] >= 0.50 && yVector[ i ] == 1.0 )
            ++numberCasesCorrect;
        else if ( pVector[ i ] < 0.50 && yVector[ i ] == 0.0 )
            ++numberCasesCorrect;
        else
            ++numberCasesWrong;
    }

    const auto total = numberCasesCorrect + numberCasesWrong;
    if ( total == 0 )
        return 0.0;
    else
        return ( 100.0 * numberCasesCorrect ) / total;
}

// ============================================================================================

TArray< float > SWLogisticRegression::ComputeBestBeta( TArray< TArray< float > > & xMatrix, TArray< float > & yVector, const int maxIterations, const float epsilon, const float jumpFactor )
{
    // Use the Newton-Raphson technique to estimate logistic regression beta parameters
    // xMatrix is a design matrix of predictor variables where the first column is augmented with all 1.0 to represent dummy x values for the b0 constant
    // yVector is a column vector of binary (0.0 or 1.0) dependent variables
    // maxIterations is the maximum number of times to iterate in the algorithm. A value of 1000 is reasonable.
    // epsilon is a closeness parameter: if all new b[i] values after an iteration are within epsilon of
    // the old b[i] values, we assume the algorithm has converged and we return. A value like 0.001 is often reasonable.
    // jumpFactor stops the algorithm if any new beta value is jumpFactor times greater than the old value. A value of 1000.0 seems reasonable.
    // The return is a column vector of the beta estimates: b[0] is the constant, b[1] for x1, etc.
    // There is a lot that can go wrong here. The algorithm involves finding a matrx inverse (see MatrixInverse) which will throw
    // if the inverse cannot be computed. The Newton-Raphson algorithm can generate beta values that tend towards infinity.
    // If anything bad happens the return is the best beta values known at the time (which could be all 0.0 values but not null).

    if ( xMatrix.Num() == 0 )
        return TArray< float >();

    const auto xRows = xMatrix.Num();
    const auto xCols = xMatrix[ 0 ].Num();

    if ( xRows != yVector.Num() )
        throw new std::exception( "The xMatrix and yVector are not compatible in LogisticRegressionNewtonParameters()" );

    // initial beta values
    TArray< float > bVector;
    bVector.Reserve( xCols );
    for ( auto i = 0; i < xCols; ++i )
    {
        bVector.Add(0.0);
    } // initialize to 0.0. TODO: consider alternatives
      //Console.WriteLine("The initial B vector is");
      //Console.WriteLine(VectorAsString(bVector)); Console.WriteLine("\n");

    // best beta values found so far
    auto bestBvector = VectorDuplicate( bVector );

    auto pVector = ConstructProbVector( xMatrix, bVector ); // a column vector of the probabilities of each row using the b[i] values and the x[i] values.
                                                            //Console.WriteLine("The initial Prob vector is: ");
                                                            //Console.WriteLine(VectorAsString(pVector)); Console.WriteLine("\n");

    //float[][] wMatrix = ConstructWeightMatrix(pVector); // deprecated. not needed if we use a shortct to comput WX. See ComputeXtilde.
    //Console.WriteLine("The initial Weight matrix is: ");
    //Console.WriteLine(MatrixAsString(wMatrix)); Console.WriteLine("\n");

    auto mse = MeanSquaredError( pVector, yVector );
    auto timesWorse = 0; // how many times are the new betas worse (i.e., give worse MSE) than the current betas

    for ( auto i = 0; i < maxIterations; ++i )
    {
        //Console.WriteLine("=================================");
        //Console.WriteLine(i);

        auto newBvector = ConstructNewBetaVector( bVector, xMatrix, yVector, pVector ); // generate new beta values using Newton-Raphson. could return null.
        if ( newBvector.Num() == 0 )
        {
            //Console.WriteLine("The ConstructNewBetaVector() helper method in LogisticRegressionNewtonParameters() returned null");
            //Console.WriteLine("because the MatrixInverse() helper method in ConstructNewBetaVector returned null");
            //Console.WriteLine("because the current (X'X~) product could not be inverted");
            //Console.WriteLine("Returning best beta vector found");
            //Console.ReadLine();
            return bestBvector;
        }

        //Console.WriteLine("New b vector is ");
        //Console.WriteLine(VectorAsString(newBvector)); Console.WriteLine("\n");

        // no significant change?
        if ( NoChange( bVector, newBvector, epsilon ) == true ) // we are done because of no significant change in beta[]
        {
            //Console.WriteLine("No significant change between old beta values and new beta values -- stopping");
            //Console.ReadLine();
            return bestBvector;
        }
        // spinning out of control?
        if ( OutOfControl( bVector, newBvector, jumpFactor ) == true ) // any new beta more than jumpFactor times greater than old?
        {
            //Console.WriteLine("The new beta vector has at least one value which changed by a factor of " + jumpFactor + " -- stopping");
            //Console.ReadLine();
            return bestBvector;
        }

        pVector = ConstructProbVector( xMatrix, newBvector );

        // are we getting worse or better?
        const auto newMSE = MeanSquaredError( pVector, yVector ); // smaller is better
        if ( newMSE > mse )                                   // new MSE is worse than current SSD
        {
            ++timesWorse; // update counter
            if ( timesWorse >= 4 )
            {
                //Console.WriteLine("The new beta vector produced worse predictions even after modification four times in a row -- stopping");
                return bestBvector;
            }
            //Console.WriteLine("The new beta vector has produced probabilities which give worse predictions -- modifying new betas to halfway between old and new");
            //Console.WriteLine("Times worse = " + timesWorse);

            bVector = VectorDuplicate( newBvector ); // update current b: old b becomes not the new b but halfway between new and old
            for ( auto k = 0; k < bVector.Num(); ++k )
            {
                bVector[ k ] = ( bVector[ k ] + newBvector[ k ] ) / 2.0;
            }
            mse = newMSE; // update current SSD (do not update best b because we don't have a new best b)
                          //Console.ReadLine();
        }
        else // new SSD is be better than old
        {
            bVector = VectorDuplicate( newBvector );  // update current b: old b becomes new b
            bestBvector = VectorDuplicate( bVector ); // update best b
            mse = newMSE;                             // update current MSE
            timesWorse = 0;                           // reset counter
        }

        //float pa = PredictiveAccuracy(xMatrix, yVector, bestBvector); // how many cases are we correctly predicting
        //Console.WriteLine("Predictive accuracy is " + pa.ToString("F4"));

        //Console.WriteLine("=================================");
        //Console.ReadLine();
    } // end main iteration loop

    //Console.WriteLine("Exceeded max iterations -- stopping");
    //Console.ReadLine();
    return bestBvector;
}

// --------------------------------------------------------------------------------------------

TArray< float > SWLogisticRegression::ConstructNewBetaVector( TArray< float > & oldBetaVector, TArray< TArray< float > > & xMatrix, TArray< float > & yVector, TArray< float > & oldProbVector )
{
    // this is the heart of the Newton-Raphson technique
    // b[t] = b[t-1] + inv(X'W[t-1]X)X'(y - p[t-1])
    //
    // b[t] is the new (time t) b column vector
    // b[t-1] is the old (time t-1) vector
    // X' is the transpose of the X matrix of x data (1.0, age, sex, chol)
    // W[t-1] is the old weight matrix
    // y is the column vector of binary dependent variable data
    // p[t-1] is the old column probability vector (computed as 1.0 / (1.0 + exp(-z) where z = b0x0 + b1x1 + . . .)

    // note: W[t-1] is nxn which could be huge so instead of computing b[t] = b[t-1] + inv(X'W[t-1]X)X'(y - p[t-1])
    // compute b[t] = b[t-1] + inv(X'X~)X'(y - p[t-1]) where X~ is W[t-1]X computed directly
    // the idea is that the vast majority of W[t-1] cells are 0.0 and so can be ignored

    auto Xt = MatrixTranspose( xMatrix );             // X'
    auto A = ComputeXtilde( oldProbVector, xMatrix ); // WX
    auto B = MatrixProduct( Xt, A );                  // X'WX

    auto C = MatrixInverse( B ); // inv(X'WX)
    if ( C.Num() == 0 )          // computing the inverse can blow up easily
        return TArray< float >();

    auto D = MatrixProduct( C, Xt );                       // inv(X'WX)X'
    auto YP = VectorSubtraction( yVector, oldProbVector ); // y-p
    auto E = MatrixVectorProduct( D, YP );                 // inv(X'WX)X'(y-p)
    auto result = VectorAddition( oldBetaVector, E );      // b + inv(X'WX)X'(y-p)

    return result; // could be null!
}

// --------------------------------------------------------------------------------------------

TArray< TArray< float > > SWLogisticRegression::ComputeXtilde( TArray< float > & pVector, TArray< TArray< float > > & xMatrix )
{
    // note: W[t-1] is nxn which could be huge so instead of computing b[t] = b[t-1] + inv(X'W[t-1]X)X'(y - p[t-1]) directly
    // we compute the W[t-1]X part, without the use of W.
    // Since W is derived from the prob vector and W has non-0.0 elements only on the diagonal we can avoid a ton of work
    // by using the prob vector directly and not computing W at all.
    // Some of the research papers refer to the product W[t-1]X as X~ hence the name of this method.
    // ex: if xMatrix is 10x4 then W would be 10x10 so WX would be 10x4 -- the same size as X

    const auto pRows = pVector.Num();
    const auto xRows = xMatrix.Num();
    const auto xCols = xMatrix[ 0 ].Num();

    if ( pRows != xRows )
        throw new std::exception( "The pVector and xMatrix are not compatible in ComputeXtilde" );

    // we are not doing marix multiplication. the p column vector sort of lays on top of each matrix column.
    auto result = MatrixCreate( pRows, xCols ); // could use (xRows, xCols) here

    for ( auto i = 0; i < pRows; ++i )
    {
        for ( auto j = 0; j < xCols; ++j )
        {
            result[ i ][ j ] = pVector[ i ] * ( 1.0 - pVector[ i ] ) * xMatrix[ i ][ j ]; // note the p(1-p)
        }
    } // i
    return result;
}

// --------------------------------------------------------------------------------------------

bool SWLogisticRegression::NoChange( TArray< float > & oldBvector, TArray< float > & newBvector, const float epsilon )
{
    // true if all new b values have changed by amount smaller than epsilon
    for ( auto i = 0; i < oldBvector.Num(); ++i )
    {
        if ( FMath::Abs( oldBvector[ i ] - newBvector[ i ] ) > epsilon ) // we have at least one change
            return false;
    }
    return true;
}

bool SWLogisticRegression::OutOfControl( TArray< float > & oldBvector, TArray< float > & newBvector, const float jumpFactor )
{
    // true if any new b is jumpFactor times greater than old b
    for ( auto i = 0; i < oldBvector.Num(); ++i )
    {
        if ( oldBvector[ i ] == 0.0 )
            return false; // if old is 0.0 anything goes for the new value

        if ( FMath::Abs( oldBvector[ i ] - newBvector[ i ] ) / FMath::Abs( oldBvector[ i ] ) > jumpFactor ) // too big a change.
            return true;
    }
    return false;
}

// --------------------------------------------------------------------------------------------

TArray< float > SWLogisticRegression::ConstructProbVector( TArray< TArray< float > > & xMatrix, TArray< float > & bVector )
{
    // p = 1 / (1 + exp(-z) where z = b0x0 + b1x1 + b2x2 + b3x3 + . . .
    // suppose X is 10 x 4 (cols are: x0 = const. 1.0, x1, x2, x3)
    // then b would be a 4 x 1 (col vecror)
    // then result of X times b is (10x4)(4x1) = (10x1) column vector

    const auto xRows = xMatrix.Num();
    const auto xCols = xMatrix[ 0 ].Num();
    const auto bRows = bVector.Num();

    if ( xCols != bRows )
        throw new std::exception( "xMatrix and bVector are not compatible in ConstructProbVector" );

    auto result = VectorCreate( xRows ); // ex: if xMatrix is size 10 x 4 and bVector is 4 x 1 then prob vector is 10 x 1 (one prob for every row of xMatrix)

    float z = 0.0;
    float p = 0.0;

    for ( auto i = 0; i < xRows; ++i )
    {
        z = 0.0;
        for ( auto j = 0; j < xCols; ++j )
        {
            z += xMatrix[ i ][ j ] * bVector[ j ]; // b0(1.0) + b1x1 + b2x2 + . . .
        }
        p = 1.0 / ( 1.0 + FMath::Exp( -z ) ); // consider checking for huge value of Math.Exp(-z) here
        result[ i ] = p;
    }
    return result;
}

// --------------------------------------------------------------------------------------------

float SWLogisticRegression::MeanSquaredError( TArray< float > & pVector, TArray< float > & yVector )
{
    // how good are the predictions? (using an already-calculated prob vector)
    // note: it is possible that a model with better (lower) MSE than a second model could give worse predictive accuracy.
    const auto pRows = pVector.Num();
    const auto yRows = yVector.Num();
    if ( pRows != yRows )
        throw new std::exception( "The prob vector and the y vector are not compatible in MeanSquaredError()" );
    if ( pRows == 0 )
        return 0.0;
    float result = 0.0;
    for ( auto i = 0; i < pRows; ++i )
    {
        result += ( pVector[ i ] - yVector[ i ] ) * ( pVector[ i ] - yVector[ i ] );
        //result += Math.Abs(pVector[i] - yVector[i]); // average absolute deviation approach
    }
    return result / pRows;
}

// --------------------------------------------------------------------------------------------

// ============================================================================================

TArray< TArray< float > > SWLogisticRegression::MatrixCreate( const int rows, const int cols )
{
    // creates a matrix initialized to all 0.0. assume rows and cols > 0
    TArray< TArray< float > > result;
    result.Reserve( rows );
    for ( auto i = 0; i < rows; ++i )
    {
        result.Add( TArray<float>() );
        result[ i ].Reserve( cols );
        for (auto j = 0; j < cols; j++)
        {
            result[i].Add(0);
        }
    } // explicit initialization not necessary.
    return result;
}

TArray< float > SWLogisticRegression::VectorCreate( const int rows )
{
    TArray< float > result;
    result.Reserve( rows ); // we use this technique when we want to make column vector creation explicit
    for (auto i = 0; i < rows; i++)
    {
        result.Add(0);
    }
    return result;
}

// FString SWLogisticRegression::MatrixAsString( TArray<TArray<float>> matrix, int numRows, int digits, int width )
// {
//     string s = "";
//     for (int i = 0; i < matrix.Length && i < numRows; ++i)
//     {
//         for (int j = 0; j < matrix[i].Length; ++j)
//         {
//             s += matrix[i][j].ToString("F" + digits).PadLeft(width) + " ";
//         }
//         s += Environment.NewLine;
//     }
//     return s;
// }

TArray< TArray< float > > SWLogisticRegression::MatrixDuplicate( TArray< TArray< float > > & matrix )
{
    // allocates/creates a duplicate of a matrix. assumes matrix is not null.
    auto result = MatrixCreate( matrix.Num(), matrix[ 0 ].Num() );
    for ( auto i = 0; i < matrix.Num(); ++i ) // copy the values
        for ( auto j = 0; j < matrix[ i ].Num(); ++j )
            result[ i ][ j ] = matrix[ i ][ j ];
    return result;
}

TArray< float > SWLogisticRegression::VectorAddition( TArray< float > & vectorA, TArray< float > &  vectorB )
{
    const auto aRows = vectorA.Num();
    const auto bRows = vectorB.Num();
    if ( aRows != bRows )
        throw new std::exception( "Non-conformable vectors in VectorAddition" );
    TArray< float > result;
    result.Reserve( aRows );
    for ( auto i = 0; i < aRows; ++i )
        result.Add(vectorA[ i ] + vectorB[ i ]);
    return result;
}

TArray< float > SWLogisticRegression::VectorSubtraction( TArray< float > & vectorA, TArray< float > & vectorB )
{
    const auto aRows = vectorA.Num();
    const auto bRows = vectorB.Num();
    if ( aRows != bRows )
        throw new std::exception( "Non-conformable vectors in VectorSubtraction" );
    TArray< float > result;
    result.Reserve( aRows );
    for ( auto i = 0; i < aRows; ++i )
        result.Add(vectorA[ i ] - vectorB[ i ]);
    return result;
}

// FString SWLogisticRegression::VectorAsString( TArray<float> vector, int count, int digits, int width )
// {
//     string s = "";
//     for (int i = 0; i < vector.Length && i < count; ++i)
//         s += " " + vector[i].ToString("F" + digits).PadLeft(width) + Environment.NewLine;
//     s += Environment.NewLine;
//     return s;
// }

TArray< float > SWLogisticRegression::VectorDuplicate( TArray< float > & vector )
{
    TArray< float > result;
    result.Reserve( vector.Num() );
    for ( auto i = 0; i < vector.Num(); ++i )
        result.Add(vector[ i ]);
    return result;
}

TArray< TArray< float > > SWLogisticRegression::MatrixTranspose( TArray< TArray< float > > & matrix )
{
    const auto rows = matrix.Num();
    const auto cols = matrix[ 0 ].Num();             // assume all columns have equal size
    auto result = MatrixCreate( cols, rows ); // note the indexing swap
    for ( auto i = 0; i < rows; ++i )
    {
        for ( auto j = 0; j < cols; ++j )
        {
            result[ j ][ i ] = matrix[ i ][ j ];
        }
    }
    return result;
}

TArray< TArray< float > > SWLogisticRegression::MatrixProduct( TArray< TArray< float > > & matrixA, TArray< TArray< float > > & matrixB )
{
    const auto aRows = matrixA.Num();
    const auto aCols = matrixA[ 0 ].Num();
    const auto bRows = matrixB.Num();
    const auto bCols = matrixB[ 0 ].Num();
    if ( aCols != bRows )
        throw new std::exception( "Non-conformable matrices in MatrixProduct" );

    auto result = MatrixCreate( aRows, bCols );

    for ( auto i = 0; i < aRows; ++i )         // each row of A
        for ( auto j = 0; j < bCols; ++j )     // each col of B
            for ( auto k = 0; k < aCols; ++k ) // could use k < bRows
                result[ i ][ j ] += matrixA[ i ][ k ] * matrixB[ k ][ j ];

    return result;
}

TArray< float > SWLogisticRegression::MatrixVectorProduct( TArray< TArray< float > > & matrix, TArray< float > & vector )
{
    const auto mRows = matrix.Num();
    const auto mCols = matrix[ 0 ].Num();
    const auto vRows = vector.Num();
    if ( mCols != vRows )
        throw new std::exception( "Non-conformable matrix and vector in MatrixVectorProduct" );
    TArray< float > result;
    result.Reserve( mRows ); // an n x m matrix times a m x 1 column vector is a n x 1 column vector
    for ( auto i = 0; i < mRows; ++i )
    {
        result.Add(0);
        for ( auto j = 0; j < mCols; ++j )
            result[ i ] += matrix[ i ][ j ] * vector[ j ];
    }
    return result;
}

TArray< TArray< float > > SWLogisticRegression::MatrixInverse( TArray< TArray< float > > & matrix )
{
    const auto n = matrix.Num();
    auto result = MatrixDuplicate( matrix );

    TArray< int > perm;
    int toggle;
    auto lum = MatrixDecompose( matrix, perm, toggle );
    if ( lum.Num() == 0 )
        return TArray< TArray< float > >();

    TArray< float > b;
    b.Reserve( n );
    for ( auto i = 0; i < n; ++i )
    {
        for ( auto j = 0; j < n; ++j )
        {
            if ( i == perm[ j ] )
                b.Add(1.0);
            else
                b.Add(0.0);
        }

        auto x = HelperSolve( lum, b ); //

        for ( auto j = 0; j < n; ++j )
            result[ j ][ i ] = x[ j ];
    }
    return result;
}

TArray< float > SWLogisticRegression::HelperSolve( TArray< TArray< float > > & luMatrix, TArray< float > & b )
{
    // solve Ax = b if you already have luMatrix from A and b has been permuted
    const auto n = luMatrix.Num();

    // 1. make a copy of the permuted b vector
    TArray< float > x;
    x.Reserve( n );
    for ( auto i = 0; i < b.Num(); ++i )
        x.Add(b[ i ]);

    // 2. solve Ly = b using forward substitution
    for ( auto i = 1; i < n; ++i )
    {
        auto sum = x[ i ];
        for ( auto j = 0; j < i; ++j )
        {
            sum -= luMatrix[ i ][ j ] * x[ j ];
        }
        x[ i ] = sum;
    }

    // 3. solve Ux = y using backward substitution
    x[ n - 1 ] /= luMatrix[ n - 1 ][ n - 1 ];
    for ( auto i = n - 2; i >= 0; --i )
    {
        auto sum = x[ i ];
        for ( auto j = i + 1; j < n; ++j )
        {
            sum -= luMatrix[ i ][ j ] * x[ j ];
        }
        x[ i ] = sum / luMatrix[ i ][ i ];
    }

    return x;
}

// -------------------------------------------------------------------------------------------------------------------

TArray< TArray< float > > SWLogisticRegression::MatrixDecompose( TArray< TArray< float > > & matrix, TArray< int > & perm, int & tog )
{
    // Doolittle's method (1.0s on L diagonal) with partial pivoting
    const auto rows = matrix.Num();
    const auto cols = matrix[ 0 ].Num(); // assume all rows have the same number of columns so just use row [0].
    if ( rows != cols )
        throw new std::exception( "Attempt to MatrixDecompose a non-square matrix" );

    const auto n = rows; // convenience

    auto result = MatrixDuplicate( matrix ); // make a copy of the input matrix

    perm.Reset( n ); // set up row permutation result
    for ( auto i = 0; i < n; ++i )
    {
        perm.Add(i);
    }

    tog = 1; // toggle tracks number of row swaps. used by MatrixDeterminant

    float aij;

    for ( auto j = 0; j < n - 1; ++j ) // each column
    {
        auto max = FMath::Abs( result[ j ][ j ] ); // find largest value in row
        auto pRow = j;
        for ( auto i = j + 1; i < n; ++i )
        {
            aij = FMath::Abs( result[ i ][ j ] );
            if ( aij > max )
            {
                max = aij;
                pRow = i;
            }
        }

        if ( pRow != j ) // if largest value not on pivot, swap rows
        {
            const auto rowPtr = result[ pRow ];
            result[ pRow ] = result[ j ];
            result[ j ] = rowPtr;

            const auto tmp = perm[ pRow ]; // and swap perm info
            perm[ pRow ] = perm[ j ];
            perm[ j ] = tmp;

            tog = -tog; // adjust the row-swap toggle
        }

        const auto ajj = result[ j ][ j ];
        if ( FMath::Abs( ajj ) < 0.00000001f )   // if diagonal after swap is zero . . .
            return TArray< TArray< float > >(); // consider a throw

        for ( auto i = j + 1; i < n; ++i )
        {
            aij = result[ i ][ j ] / ajj;
            result[ i ][ j ] = aij;
            for ( auto k = j + 1; k < n; ++k )
            {
                result[ i ][ k ] -= aij * result[ j ][ k ];
            }
        }
    } // main j loop

    return result;
}
