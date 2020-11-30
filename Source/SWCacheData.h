#pragma once

#include <CoreMinimal.h>

#include "SWCacheData.generated.h"

class USWDDAAttempt;

UCLASS()
class USWCacheData : public UObject
{
    GENERATED_BODY()
    
public:
    void Init( FString playerId, FString challengeId, int sizeLimit = 1000 );

    void addAttempt( USWDDAAttempt * attempt );

    TArray< USWDDAAttempt * > Attempts;
    FString PlayerId;
    FString ChallengeId;
    int SizeLimit = 1000;
};