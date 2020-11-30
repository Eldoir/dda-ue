#pragma once

#include <CoreMinimal.h>

#include "SWModelLR.generated.h"

UCLASS()
class SWARMS_API USWModelLR : public UObject
{
    GENERATED_BODY()

public:
    bool isUsable() const;

    void saveBetasToCsv( FString csvFile );

    //Attention : PROBA DE SUCCES, pas difficulté
    //Donner la valeur des variables en entrée (les theta)
    float Predict( TArray< float > & values );

    //trouve le bon params xi pour une proba donnée et toutes les variables xj(j!=i) fixées sauf une (sinon pas de res)
    //xi = ( (-ln(1/p -1) - (b(j!=i)x(j!=i)) ) / bi;
    //Attention : PROBA DE SUCCES, pas difficulté
    //Attention, n'écrit pas dans values !!  regarder le retour
    float InvPredict( float proba, TArray< float > values = TArray<float>(), int varToSet = 0 );

    TArray< float > Betas;
};
