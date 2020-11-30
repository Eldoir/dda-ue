#pragma once

#include <CoreMinimal.h>

#include "SWDataLR.generated.h"

UCLASS()
class SWARMS_API USWDataLR : public UObject
{
    GENERATED_BODY()

public:
    USWDataLR();

    USWDataLR * shuffle();

    void split( int pcentStartExtract, int pcentEndExtract, USWDataLR * partOut, USWDataLR * partIn );

    USWDataLR * getLastNRows( int nbRows );

    void LoadDataFromList( TArray< TArray< float > > & indepVars, TArray< float > & depVars );

    void LoadDataFromCsv( FString csvFile );

    void saveDataToCsv( FString csvFile );

    TArray< TArray< float > > IndepVar;
    TArray< float > DepVar;
};
