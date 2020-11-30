#include "SWDDAAttempt.h"

bool USWDDAAttempt::IsSame( USWDDAAttempt * other )
{
    if ( other == nullptr )
        return false;

    if ( !( other->Thetas.Num() == 0 && Thetas.Num() == 0 ) )
    {
        if ( other->Thetas.Num() == 0 )
            return false;
        if ( Thetas.Num() == 0 )
            return false;

        float delta = 0;
        for ( auto index = 0; index < Thetas.Num(); ++index )
            delta += FMath::Abs( Thetas[ index ] - other->Thetas[ index ] );
        if ( delta / Thetas.Num() > 0.00001 )
            return false;
    }

    if ( Result != other->Result )
        return false;

    return true;
}