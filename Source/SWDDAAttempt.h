#pragma once

#include <CoreMinimal.h>

#include "SWDDAAttempt.generated.h"

UCLASS(BlueprintType)
class SWARMS_API USWDDAAttempt : public UObject
{
    GENERATED_BODY()

public:
    UPROPERTY(BlueprintReadWrite)
    TArray< float > Thetas; //Variable describing challenge difficulty
    UPROPERTY(BlueprintReadWrite)
    float Result;           //1 if player won this challenge, 0 if not

    bool IsSame( USWDDAAttempt * other ); //Not using equals because dont want to mess with Equals and hashcodes, object not immutable (should be ?)
};
