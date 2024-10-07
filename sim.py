import pythia8
import pandas as pd
import math

pythia = pythia8.Pythia()
pythia.readString("SoftQCD:nonDiffractive = on")
pythia.readString("9900022:new = gamma_dark DarkPhoton")
pythia.readString("9900022:spinType = 1")
pythia.readString("9900022:chargeType = 0")
pythia.readString("9900022:colType = 0")
pythia.readString("9900022:m0 = 1e-20")
pythia.readString("9900022:isResonance = false")
pythia.readString("111:addChannel = 1 0.000001 101 22 9900022")
pythia.init()

events_num = 500000
events = []

dark_photon_events = []

for i_event in range(events_num):
    if not pythia.next():
        continue

    HT = MET_x = MET_y = 0.0
    dark_photon_produced = False
    visible_energy = visible_px = visible_py = visible_pz = 0.0

    for i in range(pythia.event.size()):
        particle = pythia.event[i]

        if particle.isFinal() and particle.idAbs() not in [12, 14, 16, 9900022]:
            HT += particle.pT()
            visible_energy += particle.e()
            visible_px += particle.px()
            visible_py += particle.py()
            visible_pz += particle.pz()

        if particle.isFinal() and particle.idAbs() in [12, 14, 16, 9900022]:
            MET_x += particle.px()
            MET_y += particle.py()

        if particle.id() == 9900022:
            dark_photon_produced = True

    MET = (MET_x**2 + MET_y**2) ** 0.5
    delta_phi = math.atan2(MET_y, MET_x) - math.atan2(visible_py, visible_px)
    delta_phi = math.atan2(math.sin(delta_phi), math.cos(delta_phi))

    M_T = math.sqrt(2 * HT * MET * (1 - math.cos(delta_phi)))
    MR = math.sqrt(max(0, (visible_energy**2 - visible_pz**2)))
    Rsq = (M_T / MR) ** 2 if MR > 0 else 0

    event_data = {
        "Event Number": i_event,
        "HT": HT,
        "MET": MET,
        "MR": MR,
        "Rsq": Rsq,
        "Dark Photon Produced": dark_photon_produced,
    }
    events.append(event_data)

    if dark_photon_produced:
        dark_photon_events.append(event_data)


# Print the event number, HT, MET, and confirm dark photon production for stored events
for data_point in dark_photon_events:
    print(data_point)

df_events = pd.DataFrame(events)
csv_file_path = "data/sim_with_razor3.csv"

try:
    with open(csv_file_path, "a") as f:
        df_events.to_csv(f, header=f.tell() == 0, index=False)
except FileNotFoundError:
    df_events.to_csv(csv_file_path, index=False)
