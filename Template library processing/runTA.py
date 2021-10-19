from TemplatesAnalyzer import TemplatesAnalyzer


DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna radial distances to shower core, in cm
SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-test/'
FIT_DIRECTORY = ['/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles17/',
                 '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles18/',
                 '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles19/']
REAS_DIRECTORY = ['/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/',
                  '/mnt/hgfs/Shared data/BulkSynth/bulksynth-18/',
                  '/mnt/hgfs/Shared data/BulkSynth/bulksynth-19/']
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramFit/'


fitter = TemplatesAnalyzer(SIM_DIRECTORY, SIM_DIRECTORY, DISTANCES, parameter_path=PARAM_DIRECTORY)
fitter.working_path = FIT_DIRECTORY
fitter.directory = REAS_DIRECTORY
fitter.fit_parameters()
