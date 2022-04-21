import hopsy
import x3c2fluxpy as x3c

#problems = {
#    "STAT-1": [(model.A.shape[1], hopsy.Problem(model.A, model.b, model, model.initial_point), "STAT-1") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal.fml")]], #+
#    "STAT-1-ni": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-1-ni") for model in [x3c.X3CModel("models/Spiralus_STAT_unimodal_ni.fml")]], #+
    #"STAT-2": [(model.A.shape[1], hopsy.Problem(model.A, model.b, model, model.initial_point), "STAT-2") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")]],
#    "STAT-2-ni": [(model.A.shape[1], hopsy.add_box_constraints(hopsy.Problem(model.A, model.b, model, model.initial_point), -1000, 1000), "STAT-2-ni") for model in [x3c.X3CModel("models/Spiralus_STAT_bimodal_ni.fml")]],
#}

m = x3c.X3CModel("models/Spiralus_STAT_bimodal.fml")

#for key, problem in problems.items():
#    print(key, problem[0][1].A.shape[-1])
    #print(key, problem[1].A.shape[-1])

