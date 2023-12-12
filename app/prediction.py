import joblib

def PCOS_predict(age,weight,height,BMI,bldgroup,weightgain,hairgrowth,skindark,hairloss,pimples,fastfood):
    model=joblib.load("/home/samuel/Documents/python/PCOS/app/PCOS_Pred.joblib")
    return model.predict([[age,weight,height,BMI,bldgroup,weightgain,hairgrowth,skindark,hairloss,pimples,fastfood]])

