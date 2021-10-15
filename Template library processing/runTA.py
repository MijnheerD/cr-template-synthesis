from TemplatesAnalyzer import TemplatesAnalyzer


DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna radial distances to shower core, in cm
SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-18/'
FIT_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles18/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramFit18/'


fitter = TemplatesAnalyzer(SIM_DIRECTORY, FIT_DIRECTORY, DISTANCES, parameter_path=PARAM_DIRECTORY)
fitter.fit_templates()
