#pragma once

#include <CoreMinimal.h>

class USWModelLR;
class USWDataLR;

class SWARMS_API SWLogisticRegression
{
public:
    //Le fichier doit contenir pour chaque lignes les valeurs des indépendants suivie de la dépendante
    static USWModelLR * ComputeModel( USWDataLR * datas );
    static float TestModel( USWModelLR * model, USWDataLR * testData );
    static float PredictiveAccuracy( TArray< TArray< float > > & xMatrix, TArray< float > & yVector, TArray< float > & bVector );
    static TArray< float > ComputeBestBeta( TArray< TArray< float > > & xMatrix, TArray< float > & yVector, int maxIterations, float epsilon, float jumpFactor );
    static TArray< float > ConstructNewBetaVector( TArray< float > & oldBetaVector, TArray< TArray< float > > & xMatrix, TArray< float > & yVector, TArray< float > & oldProbVector );
    static TArray< TArray< float > > ComputeXtilde( TArray< float > & pVector, TArray< TArray< float > > & xMatrix );
    static bool NoChange( TArray< float > & oldBvector, TArray< float > & newBvector, float epsilon );
    static bool OutOfControl( TArray< float > & oldBvector, TArray< float > & newBvector, float jumpFactor );
    static TArray< float > ConstructProbVector( TArray< TArray< float > > & xMatrix, TArray< float > & bVector );
    static float MeanSquaredError( TArray< float > & pVector, TArray< float > & yVector );
    static TArray< TArray< float > > MatrixCreate( int rows, int cols );
    static TArray< float > VectorCreate( int rows );
    //static FString MatrixAsString( TArray< TArray< float > > matrix, int numRows, int digits, int width );
    static TArray< TArray< float > > MatrixDuplicate( TArray< TArray< float > > & matrix );
    static TArray< float > VectorAddition( TArray< float > & vectorA, TArray< float > & vectorB );
    static TArray< float > VectorSubtraction( TArray< float > & vectorA, TArray< float > & vectorB );
    //static FString VectorAsString( TArray< float > vector, int count, int digits, int width );
    static TArray< float > VectorDuplicate( TArray< float > & vector );
    static TArray< TArray< float > > MatrixTranspose( TArray< TArray< float > > & matrix );
    static TArray< TArray< float > > MatrixProduct( TArray< TArray< float > > & matrixA, TArray< TArray< float > > & matrixB );
    static TArray< float > MatrixVectorProduct( TArray< TArray< float > > & matrix, TArray< float > & vector );
    static TArray< TArray< float > > MatrixInverse( TArray< TArray< float > > & matrix );
    static TArray< float > HelperSolve( TArray< TArray< float > > & luMatrix, TArray< float > & b );
    static TArray< TArray< float > > MatrixDecompose( TArray< TArray< float > > & matrix, TArray< int > & perm, int & tog );
};