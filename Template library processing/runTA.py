from TemplatesAnalyzer import TemplatesAnalyzer


DISTANCES = [1, 4000, 7500, 11000, 15000, 37500]  # antenna_nr radial distances to shower core, in cm
COLORS = ['cyan', 'magenta', 'yellow']  # colors to be used for fit figures

SIM_DIRECTORY = '/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/'
PARAM_DIRECTORY = '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/paramFit20UB50/'
FIT_DIRECTORY = ['/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles20UB5017/',
                 '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles20UB5018/',
                 '/home/mdesmet/PycharmProjects/cr-template-synthesis/Amplitude spectrum fitting/fitFiles20UB5019/']
REAS_DIRECTORY = ['/mnt/hgfs/Shared data/BulkSynth/bulksynth-17/',
                  '/mnt/hgfs/Shared data/BulkSynth/bulksynth-18/',
                  '/mnt/hgfs/Shared data/BulkSynth/bulksynth-19/']


fitter = TemplatesAnalyzer(SIM_DIRECTORY, SIM_DIRECTORY, DISTANCES, parameter_path=PARAM_DIRECTORY)
fitter.fit_center = 50
fitter.working_path = FIT_DIRECTORY
fitter.directory = REAS_DIRECTORY
fitter.fit_parameters()
