#include "SWModelLR.h"

#include <Misc/FileHelper.h>

bool USWModelLR::isUsable() const
{
    return ( Betas.Num() != 0 );
}

void USWModelLR::saveBetasToCsv( const FString csvFile )
{
    FString content;

    for ( auto index = 0; index < Betas.Num(); ++index )
    {
        content.Append( FString::SanitizeFloat( Betas[ index ] ) );
        if ( index < Betas.Num() - 1 )
            content.Append( ";" );
    }
    content.Append( "\n" );

    FFileHelper::SaveStringToFile(content, *csvFile, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
}

float USWModelLR::Predict( TArray< float > & values )
{
    // p = 1 / (1 + exp(-z) where z = b0x0 + b1x1 + b2x2 + b3x3 + . . .

    if ( Betas.Num() == 0 || values.Num() != Betas.Num() - 1 )
        throw new std::exception( "Impossible to predict, no betas yet or not good number of variables" );

    auto result = 0.f; // ex: if xMatrix is size 10 x 4 and bVector is 4 x 1 then prob vector is 10 x 1 (one prob for every row of xMatrix)

    auto z = 0.f;

    z = 1.0 * Betas[ 0 ]; // b0(1.0)
    for ( auto index = 0; index < Betas.Num() - 1; ++index )
    {
        z += values[ index ] * Betas[ index + 1 ]; // z + b1x1 + b2x2 + . . .
    }
    result = 1.0 / ( 1.0 + FMath::Exp( -z ) ); // consider checking for huge value of Math.Exp(-z) here

    return result;
}

float USWModelLR::InvPredict( const float proba, TArray< float > values, const int varToSet )
{
    auto valueXi = 0.f;
    
    if(proba > 1 || proba < 0)
    {
        //Console.WriteLine("WARNING : proba " + proba + "is not 0-1 so model is going to crash");
    }
     
    if (Betas.Num() == 0)
        return 0.0f;

    auto sommeBjXjNotI = 1.0f * Betas[0]; // b0(1.0)

    //Si une seule vairable, on fait direct la prédiction , pas besoin de bloquer les autres
    if(Betas.Num() == 2)
    {
        valueXi = ((-FMath::Log2(1.0 / proba - 1) - sommeBjXjNotI)) / Betas[1];
    }
    else
    {
        for (auto index = 0; index < Betas.Num()-1; ++index)
        {
            if (index != varToSet)
                sommeBjXjNotI += values[index] * Betas[index + 1]; // z + b1x1 + b2x2 + . . .
        }
        valueXi = ((-FMath::Log2 (1.0 / proba - 1) - sommeBjXjNotI)) / Betas[varToSet + 1];
    }
               
    return valueXi;
}
