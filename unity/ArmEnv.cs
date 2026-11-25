// ArmEnv.cs  -  '관절 3개짜리 로봇 팔로 목표 지점에 손끝 닿기' 강화학습 환경 (Unity 쪽)
// 빈 GameObject 에 붙이면 된다.  바닥/팔/목표/카메라/조명은 실행할 때 코드가 직접 만든다.
//   Python -> Unity : {"cmd":"reset","seed":123}   또는   {"cmd":"step","a0":-1..1,"a1":-1..1,"a2":-1..1}
//   Unity -> Python : {"joints":[..3],"vel":[..3],"tip":[x,y,z],"target":[x,y,z]}      (각도는 라디안)
// 상태 만들기·보상 계산은 Python(arm_env.py)에서 한다.  Unity 는 '일어난 일(사실)'만 알려 준다.
//
// 팔 구조 (Unity ArticulationBody 로 관절 물리를 계산한다)
//   Base(고정) -> 관절0 Shoulder Yaw(Y축 회전, 좌우로 돎) -> 관절1 Shoulder Pitch(어깨, 위아래) -> 관절2 Elbow(팔꿈치) -> 손끝(Tip)
//   각 관절은 '목표 각도'를 향해 스프링-댐퍼(Drive)로 힘을 낸다.  중력도 작용하므로 팔이 무게 때문에 처진다.
//   행동 = 각 관절의 목표 각도를 한 스텝에 얼마나 바꿀지 (-1 ~ 1  x  Max Delta Deg)
// 물리는 자동으로 돌리지 않고 Physics.Simulate 로 한 틱씩 직접 진행한다 (같은 seed + 같은 행동 = 같은 결과).
//
// Ticks Per Frame:  0 = 한 스텝을 한 프레임에 끝냄(학습용, 빠름),  1 = 프레임당 1틱(눈으로 볼 수 있는 속도)
//
// [집기 과제] reset 에 "task":"pick" 을 붙이면 물체(빨강/파랑) 1개와 같은 색 상자 2개가 나온다.  step 에 "a3" 추가(>0 집게 닫기, <=0 열기).
//   잡기는 단순화: 집게를 닫았을 때 손끝-물체 거리가 Grab Dist 이하면 물체가 손끝에 붙고, 열면 떨어진다.
//   떨어진 물체는 Rigidbody 로 중력·충돌(바닥, 상자 벽)을 받는다.  상자 안에 들어갔는지는 기하 검사로 알려 준다.
//   "task" 가 없으면 원래 도달 과제와 완전히 같다.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

[Serializable]
public class ArmCmd            // Python -> Unity
{
    public string cmd;
    public int seed;
    public string task;        // "pick" 이면 집기 과제, 없으면 도달 과제
    public float a0, a1, a2;
    public float a3;           // 집게 (>0 닫기)
    // reset 때만 쓰는 선택 값 (없으면 Inspector 값).  pick v2 실험용
    public float max_delta;    // 한 스텝 최대 목표 각도 변화 (도)
    public float grab_dist;    // 잡히는 거리
    public int rand_pose;      // 1: 판을 무작위 자세에서 시작 (v2 학습)
}

[Serializable]
public class ArmState          // Unity -> Python
{
    public float[] joints;     // 관절 각도 (라디안)
    public float[] vel;        // 관절 각속도 (라디안/초)
    public float[] tip;        // 손끝 위치 (월드 좌표)
    public float[] target;     // 목표 위치
    // ---- 집기 과제 ----
    public float[] obj;        // 물체 위치
    public float[] boxes;      // 상자 0(빨강), 1(파랑) 바닥 중심 위치  [x0,y0,z0,x1,y1,z1]
    public int held;           // 1: 손끝에 붙어 있음
    public int grip;           // 1: 집게 닫힘
    public int color;          // 물체 색 0 빨강, 1 파랑
    public int in_box;         // 물체가 들어 있는 상자 번호 (-1: 없음)
}

public class ArmEnv : MonoBehaviour
{
    [Header("Network")]
    public string host = "127.0.0.1";
    public int port = 9999;

    [Header("Simulation")]
    public int ticksPerStep = 10;        // 한 스텝 = 10틱 = 0.1초
    public int ticksPerFrame = 0;        // 0: 학습용(빠름),  1: 눈으로 보기
    public bool realTime = false;        // 켜면 Ticks Per Frame 대신 실제 시계에 맞춰 진행 (정확히 1배속, 영상 녹화용)
    public float tickDt = 0.01f;

    [Header("Arm (Python arm_env.py 의 값과 같아야 함)")]
    public float upperLength = 1.4f;     // 윗팔 길이
    public float foreLength = 1.2f;      // 아래팔 길이
    public float yawLimit = 150f;        // 관절0 회전 범위 (도, ±)
    public float shoulderLimit = 100f;   // 관절1
    public float elbowLimit = 120f;      // 관절2
    public float maxDeltaDeg = 8f;       // 행동 1 이 한 스텝에 바꾸는 목표 각도
    public float stiffness = 3000f;      // 관절 스프링 세기
    public float damping = 300f;         // 관절 댐퍼
    public float forceLimit = 600f;      // 관절이 낼 수 있는 최대 토크
    public float linkMass = 1f;

    [Header("Target")]
    public float successDist = 0.2f;     // 표시용(성공 판정은 Python 이 한다)

    [Header("Pick (Python pick_env.py 의 값과 같아야 함)")]
    public float grabDist = 0.3f;        // 집게를 닫을 때 손끝-물체 거리가 이 이하면 잡힘
    public float objSize = 0.24f;        // 물체(정육면체) 한 변
    public float holdOffset = 0.15f;     // 잡은 물체를 손끝에서 팔 방향으로 얼마나 앞에 둘지
    public float boxRadius = 1.7f;       // 상자 중심의 받침 기준 거리 (상자 0 은 +Z, 상자 1 은 -Z 쪽)
    public float boxInner = 0.35f;       // 상자 안쪽 반 너비
    public float boxHeight = 0.35f;      // 상자 벽 높이

    [Header("Conveyor (task \"conveyor\": 물체가 계속 나오는 연속 분류 시연)")]
    public float beltSpeed = 0.3f;       // 벨트 속도 (m/s, 팔 쪽으로)
    public float spawnInterval = 3.5f;   // 새 물체가 나오는 간격 (초)
    public float stopX = 1.45f;          // 물체가 멈추는 자리(중심 x).  pick 학습 범위(정면 1.3 ~ 2.0) 안
    public float beltEndX = 4.8f;        // 벨트 시작(물체가 나오는 쪽) x
    public int poolSize = 12;            // 돌려 쓰는 물체 수 (오래된 것부터 다시 벨트에 올림)
    public int maxOnBelt = 2;            // 벨트 위에 동시에 있을 수 있는 물체 수 (넘치면 새로 안 나옴)
    public float laneHalf = 0.36f;       // 난간 안쪽 반 너비 (물체가 나오는 자리는 이 안에서 좌우로 흩어짐)
    public int maxSameColor = 2;         // 같은 색이 이보다 많이 연달아 나오지 않게
    public int maxWaitTicks = 3000;      // 다음 물체를 기다리는 최대 틱

    // ---- 물리 오브젝트 ----
    ArticulationBody[] joint = new ArticulationBody[3];
    float[] limit = new float[3];
    float[] tgtDeg = new float[3];
    Transform tipT, targetT;
    Vector3 shoulderPos;

    // ---- 집기 과제 ----
    bool pickMode;
    bool held, gripClosed;
    int objColor;
    Rigidbody objRb;
    Transform objT;
    Collider objCol;
    Collider[] armCols;
    Transform[] boxT = new Transform[2];
    Transform[] fingerT = new Transform[2];
    GameObject gripperGO;                // 손목·손바닥·손가락 (모양만, 집기 과제에서만 보임)
    Renderer tipRend;                    // 도달 과제의 노란 손끝 구
    static readonly Color[] PickColors = { new Color(0.95f, 0.25f, 0.25f), new Color(0.25f, 0.45f, 1f) };

    // ---- 컨베이어 ----  (objRb/objT/objCol/objColor 는 '지금 집을 물체'를 가리킨다)
    bool conveyorMode, waiting;
    int waitTicks, nextSpawn, current, sorted, missorted;
    string waitReason = "";
    int lastColor = -1, colorStreak;

    // ---- reset 때 정해지는 값 ----
    float curMaxDelta, curGrabDist;
    bool randPose;
    float spawnTimer;
    System.Random convRng;
    GameObject beltRoot;
    Rigidbody beltRb;
    Vector3 beltHome;
    Rigidbody[] pool;
    int[] poolColor;

    // ---- 진행 ----
    bool stepping;
    int ticksRemaining;
    int episodes, reached;

    // ---- 통신 ----
    TcpClient client;
    StreamReader reader;
    StreamWriter writer;
    Thread connThread;
    volatile bool connected;
    volatile bool running = true;
    readonly ConcurrentQueue<string> inbox = new ConcurrentQueue<string>();
    int prevVSync, prevTargetFps;
    Vector3 prevGravity;
    bool settingsSaved;
#if UNITY_2022_2_OR_NEWER
    SimulationMode prevSimMode;
#else
    bool prevAutoSim;
#endif

    void Start()
    {
        limit[0] = yawLimit; limit[1] = shoulderLimit; limit[2] = elbowLimit;
        curMaxDelta = maxDeltaDeg;
        curGrabDist = grabDist;
        BuildScene();

        prevVSync = QualitySettings.vSyncCount;
        prevTargetFps = Application.targetFrameRate;
        prevGravity = Physics.gravity;
        settingsSaved = true;
        if (ticksPerFrame <= 0)
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
        }
        Application.runInBackground = true;
        Physics.gravity = new Vector3(0f, -9.81f, 0f);
#if UNITY_2022_2_OR_NEWER
        prevSimMode = Physics.simulationMode;
        Physics.simulationMode = SimulationMode.Script;
#else
        prevAutoSim = Physics.autoSimulation;
        Physics.autoSimulation = false;
#endif

        ResetEpisode(0);
        connThread = new Thread(ConnectLoop);
        connThread.IsBackground = true;
        connThread.Start();
    }

    // ------------------------------------------------------------------ 장면 만들기
    static Material MakeMat(Color c, float metallic = 0f, float gloss = 0.5f)
    {
        Shader sh = Shader.Find("Standard");
        if (sh == null) sh = Shader.Find("Sprites/Default");
        Material m = new Material(sh);
        m.color = c;
        m.SetFloat("_Metallic", metallic);
        m.SetFloat("_Glossiness", gloss);
        return m;
    }

    GameObject MakeMesh(string name, PrimitiveType type, Transform parent, Vector3 localPos, Vector3 scale, Color c, bool keepCollider)
    {
        GameObject go = GameObject.CreatePrimitive(type);
        go.name = name;
        go.GetComponent<Renderer>().material = MakeMat(c);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        go.transform.localScale = scale;
        if (!keepCollider)
        {
            Collider col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);
        }
        return go;
    }

    // 모양만 있는 부품.  충돌체를 바로 지워서 관절체의 질량 중심·관성(충돌체로 계산됨)에 영향을 주지 않는다
    // -> 학습한 팔과 물리적으로 똑같고 겉모습만 바뀐다
    GameObject Deco(string name, PrimitiveType type, Transform parent, Vector3 localPos, Vector3 scale, Color c,
                    float metallic = 0.2f, float gloss = 0.6f, Vector3 euler = default)
    {
        GameObject go = GameObject.CreatePrimitive(type);
        DestroyImmediate(go.GetComponent<Collider>());
        go.name = name;
        go.GetComponent<Renderer>().material = MakeMat(c, metallic, gloss);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        go.transform.localRotation = Quaternion.Euler(euler);
        go.transform.localScale = scale;
        return go;
    }

    static readonly Color ArmOrange = new Color(0.95f, 0.47f, 0.08f), Graphite = new Color(0.16f, 0.17f, 0.2f),
                          Steel = new Color(0.7f, 0.72f, 0.76f), Rubber = new Color(0.08f, 0.08f, 0.09f);

    ArticulationBody MakeJoint(string name, Transform parent, Vector3 localPos, Quaternion axisRot, float limitDeg)
    {
        GameObject go = new GameObject(name);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        ArticulationBody ab = go.AddComponent<ArticulationBody>();
        ab.jointType = ArticulationJointType.RevoluteJoint;
        ab.anchorRotation = axisRot;                       // 회전축 = 이 방향의 X 축
        ab.mass = linkMass;
        ab.twistLock = ArticulationDofLock.LimitedMotion;
        ArticulationDrive d = new ArticulationDrive();
        d.lowerLimit = -limitDeg;
        d.upperLimit = limitDeg;
        d.stiffness = stiffness;
        d.damping = damping;
        d.forceLimit = forceLimit;
        d.target = 0f;
        d.targetVelocity = 0f;
        ab.xDrive = d;
        return ab;
    }

    void BuildScene()
    {
        Camera cam = Camera.main;
        if (cam == null)
        {
            GameObject cg = new GameObject("Main Camera");
            cg.tag = "MainCamera";
            cam = cg.AddComponent<Camera>();
        }
        cam.orthographic = false;
        cam.fieldOfView = 50f;
        cam.transform.position = new Vector3(3.6f, 2.8f, -4.6f);
        cam.transform.LookAt(new Vector3(0f, 1.0f, 0f));
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.1f, 0.12f, 0.16f);

        Light l = FindObjectOfType<Light>();
        if (l == null)
        {
            GameObject lg = new GameObject("Light");
            l = lg.AddComponent<Light>();
            l.type = LightType.Directional;
            l.intensity = 1.1f;
            lg.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }
        l.shadows = LightShadows.Soft;                                     // 그림자 (보기용)

        Transform root = new GameObject("ArmWorld").transform;

        // 바닥 (윗면이 y=0)
        MakeMesh("Floor", PrimitiveType.Cube, root, new Vector3(0f, -0.1f, 0f), new Vector3(10f, 0.2f, 10f), new Color(0.25f, 0.28f, 0.33f), true);

        // 받침(고정, 루트 관절체)
        GameObject baseGO = new GameObject("ArmBase");
        baseGO.transform.SetParent(root, false);
        baseGO.transform.position = new Vector3(0f, 0.2f, 0f);
        ArticulationBody rootAb = baseGO.AddComponent<ArticulationBody>();
        rootAb.immovable = true;
        // ---- 겉모습 (모두 Deco: 충돌체 없음) ----
        Deco("BasePlate", PrimitiveType.Cylinder, baseGO.transform, new Vector3(0f, -0.18f, 0f), new Vector3(0.95f, 0.02f, 0.95f), Graphite, 0.5f, 0.4f);
        Deco("BasePedestal", PrimitiveType.Cylinder, baseGO.transform, Vector3.zero, new Vector3(0.62f, 0.2f, 0.62f), Graphite, 0.6f, 0.55f);
        Deco("BaseRing", PrimitiveType.Cylinder, baseGO.transform, new Vector3(0f, 0.17f, 0f), new Vector3(0.68f, 0.03f, 0.68f), Steel, 0.8f, 0.7f);

        // 관절0: 어깨 좌우 회전 (Y 축).  X 축을 Y 축으로 돌리는 회전 = Z 로 90도
        joint[0] = MakeJoint("JointYaw", baseGO.transform, new Vector3(0f, 0.2f, 0f), Quaternion.Euler(0f, 0f, 90f), yawLimit);
        Deco("Turntable", PrimitiveType.Cylinder, joint[0].transform, new Vector3(0f, 0.05f, 0f), new Vector3(0.54f, 0.05f, 0.54f), ArmOrange, 0.1f, 0.65f);
        for (int s = -1; s <= 1; s += 2)                                   // 어깨를 받치는 양쪽 지지판
            Deco("ShoulderYoke", PrimitiveType.Cube, joint[0].transform, new Vector3(0f, 0.15f, s * 0.2f), new Vector3(0.32f, 0.3f, 0.06f), ArmOrange, 0.1f, 0.65f);

        // 관절1: 어깨 위아래 (Z 축).  X 축을 Z 축으로 돌리는 회전 = Y 로 -90도
        joint[1] = MakeJoint("JointShoulder", joint[0].transform, new Vector3(0f, 0.2f, 0f), Quaternion.Euler(0f, -90f, 0f), shoulderLimit);
        GameObject upper = MakeMesh("UpperArm", PrimitiveType.Cube, joint[1].transform, new Vector3(0f, upperLength / 2f, 0f), new Vector3(0.22f, upperLength, 0.22f), new Color(0.3f, 0.8f, 1f), true);
        upper.GetComponent<Renderer>().enabled = false;                    // 충돌체는 그대로 두고 모양만 아래 Deco 로 바꾼다
        Deco("ShoulderHousing", PrimitiveType.Cylinder, joint[1].transform, Vector3.zero, new Vector3(0.34f, 0.17f, 0.34f), Graphite, 0.5f, 0.6f, new Vector3(90f, 0f, 0f));
        Deco("ShoulderCap", PrimitiveType.Cylinder, joint[1].transform, Vector3.zero, new Vector3(0.2f, 0.18f, 0.2f), Steel, 0.8f, 0.7f, new Vector3(90f, 0f, 0f));
        Deco("UpperLink", PrimitiveType.Capsule, joint[1].transform, new Vector3(0f, upperLength / 2f, 0f), new Vector3(0.24f, upperLength / 2f, 0.26f), ArmOrange, 0.1f, 0.65f);

        // 관절2: 팔꿈치 (Z 축)
        joint[2] = MakeJoint("JointElbow", joint[1].transform, new Vector3(0f, upperLength, 0f), Quaternion.Euler(0f, -90f, 0f), elbowLimit);
        GameObject fore = MakeMesh("ForeArm", PrimitiveType.Cube, joint[2].transform, new Vector3(0f, foreLength / 2f, 0f), new Vector3(0.18f, foreLength, 0.18f), new Color(0.4f, 0.9f, 0.6f), true);
        fore.GetComponent<Renderer>().enabled = false;
        Deco("ElbowHousing", PrimitiveType.Cylinder, joint[2].transform, Vector3.zero, new Vector3(0.3f, 0.16f, 0.3f), Graphite, 0.5f, 0.6f, new Vector3(90f, 0f, 0f));
        Deco("ElbowCap", PrimitiveType.Cylinder, joint[2].transform, Vector3.zero, new Vector3(0.17f, 0.17f, 0.17f), Steel, 0.8f, 0.7f, new Vector3(90f, 0f, 0f));
        Deco("ForeLink", PrimitiveType.Capsule, joint[2].transform, new Vector3(0f, foreLength / 2f - 0.04f, 0f), new Vector3(0.19f, foreLength / 2f - 0.04f, 0.2f), ArmOrange, 0.1f, 0.65f);
        Deco("Wrist", PrimitiveType.Cylinder, joint[2].transform, new Vector3(0f, foreLength - 0.06f, 0f), new Vector3(0.2f, 0.05f, 0.2f), Graphite, 0.6f, 0.6f);
        armCols = new Collider[] { upper.GetComponent<Collider>(), fore.GetComponent<Collider>() };

        // 손끝 표시 (위치 기록용)
        GameObject tip = MakeMesh("Tip", PrimitiveType.Sphere, joint[2].transform, new Vector3(0f, foreLength, 0f), new Vector3(0.2f, 0.2f, 0.2f), new Color(1f, 0.9f, 0.2f), false);
        tipT = tip.transform;
        tipRend = tip.GetComponent<Renderer>();

        // 목표 (충돌 없음)
        GameObject tgt = MakeMesh("Target", PrimitiveType.Sphere, root, new Vector3(1.5f, 1.2f, 0f), new Vector3(0.36f, 0.36f, 0.36f), new Color(1f, 0.3f, 0.3f), false);
        targetT = tgt.transform;

        shoulderPos = new Vector3(0f, 0.6f, 0f);
        BuildPickObjects(root);
    }

    // 집기 과제용: 집게 손가락 2개(모양만), 상자 2개(벽은 충돌 있음), 물체 1개(Rigidbody)
    void BuildPickObjects(Transform root)
    {
        // 집게 (모양만).  잡기 판정은 손끝~물체 거리로 하므로 손가락 모양은 학습과 상관없다.
        // 잡은 물체는 손끝에서 holdOffset 앞에 붙으므로 손가락 패드를 그 높이에 둔다
        gripperGO = new GameObject("Gripper");
        Transform g = gripperGO.transform;
        g.SetParent(joint[2].transform, false);
        Deco("Flange", PrimitiveType.Cylinder, g, new Vector3(0f, foreLength - 0.01f, 0f), new Vector3(0.26f, 0.02f, 0.26f), Steel, 0.8f, 0.7f);
        Deco("Palm", PrimitiveType.Cube, g, new Vector3(0f, foreLength + 0.02f, 0f), new Vector3(0.56f, 0.06f, 0.16f), Graphite, 0.5f, 0.55f);
        Deco("Rail", PrimitiveType.Cube, g, new Vector3(0f, foreLength + 0.055f, 0f), new Vector3(0.5f, 0.012f, 0.05f), Steel, 0.8f, 0.7f);
        for (int s = 0; s < 2; s++)
        {
            float inward = s == 0 ? -1f : 1f;                              // 손가락 0 은 +X 쪽, 1 은 -X 쪽
            Transform f = new GameObject("Finger" + s).transform;
            f.SetParent(g, false);
            Deco("Slider", PrimitiveType.Cube, f, new Vector3(0f, 0.01f, 0f), new Vector3(0.07f, 0.05f, 0.14f), Steel, 0.8f, 0.7f);
            Deco("Plate", PrimitiveType.Cube, f, new Vector3(0f, 0.12f, 0f), new Vector3(0.035f, 0.22f, 0.12f), Steel, 0.85f, 0.75f);
            Deco("Pad", PrimitiveType.Cube, f, new Vector3(inward * 0.027f, holdOffset - FingerBaseY, 0f), new Vector3(0.02f, 0.12f, 0.1f), Rubber, 0f, 0.2f);
            Deco("Tip", PrimitiveType.Cube, f, new Vector3(inward * 0.012f, 0.235f, 0f), new Vector3(0.05f, 0.03f, 0.12f), Steel, 0.85f, 0.75f);
            fingerT[s] = f;
        }
        SetFingers(false);

        float t = 0.06f, w = boxInner * 2f + t * 2f;
        for (int i = 0; i < 2; i++)
        {
            Transform b = new GameObject("Box" + i).transform;
            b.SetParent(root, false);
            b.position = new Vector3(0f, 0f, i == 0 ? boxRadius : -boxRadius);
            Color c = PickColors[i] * 0.8f;
            MakeMesh("Bottom", PrimitiveType.Cube, b, new Vector3(0f, 0.02f, 0f), new Vector3(w, 0.04f, w), c, true);
            MakeMesh("WallXp", PrimitiveType.Cube, b, new Vector3(boxInner + t / 2f, boxHeight / 2f, 0f), new Vector3(t, boxHeight, w), c, true);
            MakeMesh("WallXn", PrimitiveType.Cube, b, new Vector3(-boxInner - t / 2f, boxHeight / 2f, 0f), new Vector3(t, boxHeight, w), c, true);
            MakeMesh("WallZp", PrimitiveType.Cube, b, new Vector3(0f, boxHeight / 2f, boxInner + t / 2f), new Vector3(w, boxHeight, t), c, true);
            MakeMesh("WallZn", PrimitiveType.Cube, b, new Vector3(0f, boxHeight / 2f, -boxInner - t / 2f), new Vector3(w, boxHeight, t), c, true);
            boxT[i] = b;
            b.gameObject.SetActive(false);
        }

        // 물체: pick 과제는 0 번만, 컨베이어는 전부 돌려 쓴다
        pool = new Rigidbody[Mathf.Max(1, poolSize)];
        poolColor = new int[pool.Length];
        for (int i = 0; i < pool.Length; i++)
        {
            GameObject o = MakeMesh("Object" + i, PrimitiveType.Cube, root, new Vector3(1.5f, objSize / 2f, 0f), Vector3.one * objSize, PickColors[0], true);
            pool[i] = o.AddComponent<Rigidbody>();
            pool[i].mass = 0.3f;
            o.SetActive(false);
        }
        SetCurrent(0);

        // 컨베이어: 벨트(움직이는 척하는 kinematic 강체) + 끝의 멈춤판 + 양옆 난간.  모두 충돌 있음
        beltRoot = new GameObject("Conveyor");
        beltRoot.transform.SetParent(root, false);
        float stopFace = stopX - objSize / 2f - 0.01f;                    // 멈춤판 안쪽 면
        float len = beltEndX - stopFace + 0.2f, cx = stopFace - 0.1f + len / 2f;
        Color dark = new Color(0.15f, 0.15f, 0.17f), metal = new Color(0.6f, 0.62f, 0.66f);
        float lane = laneHalf;                                            // 난간 안쪽 면까지
        GameObject belt = MakeMesh("Belt", PrimitiveType.Cube, beltRoot.transform, new Vector3(cx, 0.025f, 0f), new Vector3(len, 0.05f, lane * 2f + 0.08f), dark, true);
        beltRb = belt.AddComponent<Rigidbody>();
        beltRb.isKinematic = true;
        beltHome = belt.transform.position;
        MakeMesh("Stopper", PrimitiveType.Cube, beltRoot.transform, new Vector3(stopFace - 0.04f, 0.1f, 0f), new Vector3(0.08f, 0.2f, lane * 2f + 0.08f), metal, true);
        MakeMesh("RailL", PrimitiveType.Cube, beltRoot.transform, new Vector3(cx, 0.08f, lane + 0.02f), new Vector3(len, 0.16f, 0.04f), metal, true);
        MakeMesh("RailR", PrimitiveType.Cube, beltRoot.transform, new Vector3(cx, 0.08f, -lane - 0.02f), new Vector3(len, 0.16f, 0.04f), metal, true);
        beltRoot.SetActive(false);
    }

    void SetCurrent(int i)
    {
        current = i;
        objRb = pool[i];
        objT = pool[i].transform;
        objCol = pool[i].GetComponent<Collider>();
        objColor = poolColor[i];
    }

    void PlaceObject(int i, Vector3 p, int color)
    {
        Rigidbody rb = pool[i];
        rb.gameObject.SetActive(true);
        poolColor[i] = color;
        rb.GetComponent<Renderer>().material.color = PickColors[color];
        rb.isKinematic = false;
        rb.position = p;
        rb.transform.position = p;
        rb.rotation = Quaternion.identity;
        rb.transform.rotation = Quaternion.identity;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.GetComponent<Collider>().enabled = true;
        // 팔과 물체는 서로 밀지 않는다 (비활성화하면 무시 설정이 풀리므로 매번 다시 건다)
        Collider oc = rb.GetComponent<Collider>();
        foreach (Collider c in armCols) Physics.IgnoreCollision(oc, c, true);
    }

    // 손가락 위치 (모양만).  닫으면 패드 안쪽 면이 물체 옆면(objSize/2)에 닿는다
    const float FingerBaseY = 0.04f;
    void SetFingers(bool closed)
    {
        float x = closed ? objSize / 2f + 0.037f : objSize / 2f + 0.12f;
        for (int s = 0; s < 2; s++)
            fingerT[s].localPosition = new Vector3(s == 0 ? x : -x, foreLength + FingerBaseY, 0f);
    }

    // 물체가 들어 있는 상자 번호 (-1: 없음).  들고 있으면 -1
    int ObjInBox()
    {
        if (!pickMode || held) return -1;
        for (int i = 0; i < 2; i++)
        {
            Vector3 d = objT.position - boxT[i].position;
            if (Mathf.Abs(d.x) < boxInner && Mathf.Abs(d.z) < boxInner && d.y < boxHeight) return i;
        }
        return -1;
    }

    void FollowTip()
    {
        Vector3 p = tipT.position + tipT.up * holdOffset;
        objRb.position = p;
        objT.position = p;
        objRb.rotation = tipT.rotation;
        objT.rotation = tipT.rotation;
    }

    // ------------------------------------------------------------------ 통신
    void ConnectLoop()
    {
        bool warned = false;
        while (running)
        {
            try
            {
                TcpClient c = new TcpClient(host, port);
                NetworkStream stream = c.GetStream();
                reader = new StreamReader(stream, new UTF8Encoding(false));
                writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true, NewLine = "\n" };
                client = c;
                connected = true;
                warned = false;
                Debug.Log("[Arm] Python 서버에 연결되었습니다.");
                while (running)
                {
                    string line = reader.ReadLine();
                    if (line == null) break;
                    if (line.Length > 0) inbox.Enqueue(line);
                }
            }
            catch (Exception)
            {
                if (!warned) { Debug.Log("[Arm] Python 서버를 기다리는 중... (Python 을 먼저 실행하세요)"); warned = true; }
            }
            if (connected) Debug.Log("[Arm] 연결이 종료되었습니다. 다시 접속을 기다립니다.");
            connected = false;
            try { if (client != null) client.Close(); } catch (Exception) { }
            Thread.Sleep(500);
        }
    }

    void Update()
    {
        if (!connected)
        {
            while (inbox.TryDequeue(out string _)) { }
            stepping = false;
            waiting = false;
            ticksRemaining = 0;
        }
        if (!stepping && inbox.TryDequeue(out string line))
            Handle(line);
        if (stepping)
            RunTicks();
    }

    void Handle(string json)
    {
        ArmCmd cmd = null;
        try { cmd = JsonUtility.FromJson<ArmCmd>(json); }
        catch (Exception e) { Debug.LogWarning("[Arm] JSON 해석 실패: " + e.Message); }
        if (cmd == null) { SendState(); return; }

        if (cmd.cmd == "reset")
        {
            curMaxDelta = cmd.max_delta > 0f ? cmd.max_delta : maxDeltaDeg;
            curGrabDist = cmd.grab_dist > 0f ? cmd.grab_dist : grabDist;
            randPose = cmd.rand_pose == 1;
        }
        if (cmd.cmd == "reset" && cmd.task == "conveyor")
        {
            StartConveyorWait(cmd.seed);         // 다음 물체가 멈춤판에 올 때까지 진행한 뒤 상태를 보낸다
        }
        else if (cmd.cmd == "reset")
        {
            pickMode = cmd.task == "pick";
            conveyorMode = false;
            ResetEpisode(cmd.seed);
            SendState();
        }
        else if (cmd.cmd == "step")
        {
            float[] a = { cmd.a0, cmd.a1, cmd.a2 };
            for (int i = 0; i < 3; i++)
            {
                tgtDeg[i] = Mathf.Clamp(tgtDeg[i] + Mathf.Clamp(a[i], -1f, 1f) * curMaxDelta, -limit[i], limit[i]);
                SetDriveTarget(i, tgtDeg[i]);
            }
            if (pickMode) ApplyGrip(cmd.a3 > 0f);
            ticksRemaining = Mathf.Max(1, ticksPerStep);
            stepping = true;
        }
        else
        {
            Debug.LogWarning("[Arm] 알 수 없는 명령: " + cmd.cmd);
            SendState();
        }
    }

    float tickDebt;                      // realTime: 아직 계산하지 못한 실제 시간 (초)

    void RunTicks()
    {
        int n = ticksPerFrame > 0 ? ticksPerFrame : int.MaxValue;
        if (realTime)
        {
            // 화면 주사율과 상관없이 지난 실제 시간만큼만 틱을 돌린다.  남은 시간은 다음 프레임으로 넘기고,
            // Python 응답이 늦어 밀린 시간은 0.1초까지만 따라잡는다 (한꺼번에 튀지 않게)
            tickDebt = Mathf.Min(tickDebt + Time.deltaTime, 0.1f);
            n = Mathf.FloorToInt(tickDebt / tickDt);
            tickDebt -= n * tickDt;
        }
        while (n > 0 && stepping)
        {
            Tick();
            n--;
            if (waiting)
            {
                if (++waitTicks >= maxWaitTicks || ConveyorReady()) { waiting = false; stepping = false; SendState(); }
            }
            else if (--ticksRemaining <= 0) { stepping = false; SendState(); }
        }
    }

    void Tick()
    {
        if (conveyorMode) { DriveBelt(); SpawnTick(); }
        Physics.Simulate(tickDt);
        if (held) FollowTip();
    }

    // ------------------------------------------------------------------ 컨베이어
    // 벨트 자체는 제자리.  매 틱 '뒤로 순간이동 -> 원래 자리로 MovePosition' 하면 물리엔진은 벨트가 앞으로 움직였다고 보고
    // 위에 놓인 물체를 마찰로 끌고 간다.  물체끼리 밀고, 멈춤판에 막혀 줄을 서는 것도 물리 계산 결과다.
    void DriveBelt()
    {
        beltRb.position = beltHome + Vector3.right * beltSpeed * tickDt;
        beltRb.MovePosition(beltHome);
    }

    void SpawnTick()
    {
        spawnTimer += tickDt;
        if (spawnTimer < spawnInterval) return;
        // 옆으로 흩는 폭은 물체 크기의 절반 미만: 두 물체가 멈춤판 앞에 나란히 서면 어느 것이 '맨 앞'인지 애매해지고
        // 서로 기대어 기울어진다.  한 줄로 오되 위치는 조금씩 다르게
        float zSpread = Mathf.Min(objSize * 0.4f, Mathf.Max(0f, laneHalf - objSize / 2f - 0.04f));
        Vector3 p = new Vector3(beltEndX - 0.2f, 0.05f + objSize / 2f + 0.02f, ((float)convRng.NextDouble() * 2f - 1f) * zSpread);
        int onBeltCount = 0;
        for (int i = 0; i < pool.Length; i++)
        {
            Rigidbody rb = pool[i];
            if (!rb.gameObject.activeSelf) continue;
            if ((rb.position - p).magnitude < objSize * 2f) return;                       // 나오는 자리가 막힘
            if (!(i == current && held) && rb.position.x > stopX - 0.4f && Mathf.Abs(rb.position.z) < laneHalf + 0.1f && rb.position.y < 0.6f)
                onBeltCount++;
        }
        if (onBeltCount >= maxOnBelt) return;                                             // 벨트가 꽉 참
        // 돌려 쓸 물체: 안 쓴 것 -> 상자에 들어가 있는 오래된 것 순서 (벨트 위·손에 든 물체는 건너뜀)
        for (int k = 0; k < pool.Length; k++)
        {
            int i = (nextSpawn + k) % pool.Length;
            Vector3 q = pool[i].position;
            bool onBelt = pool[i].gameObject.activeSelf && q.x > stopX - 0.4f && Mathf.Abs(q.z) < laneHalf + 0.1f;
            if ((i == current && !waiting) || onBelt) continue;
            int color = convRng.Next(2);
            if (color == lastColor && colorStreak >= maxSameColor) color = 1 - color;    // 같은 색이 너무 길게 이어지지 않게
            colorStreak = color == lastColor ? colorStreak + 1 : 1;
            lastColor = color;
            PlaceObject(i, p, color);
            nextSpawn = (i + 1) % pool.Length;
            spawnTimer = 0f;
            return;
        }
    }

    // 멈춤판 앞에 멈춰 선 물체가 있고 팔이 제자리로 돌아왔으면 시작
    bool ConveyorReady()
    {
        waitReason = "팔 복귀 중";
        for (int i = 0; i < 3; i++)
            if (Mathf.Abs(joint[i].jointPosition[0]) > 0.1f) return false;
        waitReason = "멈춘 물체 없음";
        int best = -1;
        for (int i = 0; i < pool.Length; i++)
        {
            Rigidbody rb = pool[i];
            if (!rb.gameObject.activeSelf) continue;
            Vector3 q = rb.position;
            // 멈춤판 근처(stopX + 0.1 안)에 와야 시작.  속도만 보면 느린 벨트(0.3) 위에서 아직 굴러오는 물체도
            // '멈춤'으로 판정되어 판이 시작된 뒤에도 물체가 움직였다.  너무 좁게(0.03) 잡으면 비스듬히 선 큐브는
            // 중심이 멈춤판에서 더 멀어서(최대 +0.05) 영영 시작하지 못한다
            if (!(q.x < stopX + 0.1f && q.x > stopX - 0.4f && Mathf.Abs(q.z) < laneHalf && q.y < 0.4f && rb.velocity.magnitude < 0.5f))
                continue;
            // 기울거나 다른 물체 위에 얹힌 물체는 학습 때 본 적 없는 상태 -> 멈춰 선 것을 확인한 뒤 불량품으로 치운다
            Transform t = rb.transform;
            float upright = Mathf.Max(Mathf.Abs(t.up.y), Mathf.Abs(t.right.y), Mathf.Abs(t.forward.y));
            if (upright < 0.96f || q.y > 0.05f + objSize / 2f + 0.08f)
            {
                if (rb.velocity.magnitude < 0.05f && rb.angularVelocity.magnitude < 0.1f) rb.gameObject.SetActive(false);
                continue;
            }
            if (best < 0 || q.x < pool[best].position.x)
                best = i;
        }
        if (best < 0) return false;
        SetCurrent(best);
        // 벨트 밖 물체(상자에 넣은 것·떨어뜨린 것)는 치운다.  상자에 쌓이면 다음 물체가 그 위에 얹혀
        // 상자 벽보다 높아지고(큐브 2개 = 0.48 > boxHeight) '상자 안'으로 판정되지 않아 시간초과가 난다
        for (int i = 0; i < pool.Length; i++)
        {
            Vector3 q = pool[i].position;
            if (pool[i].gameObject.activeSelf && !(q.x > stopX - 0.4f && Mathf.Abs(q.z) < laneHalf + 0.1f))
                pool[i].gameObject.SetActive(false);
        }
        return true;
    }

    void StartConveyorWait(int seed)
    {
        if (!conveyorMode)                       // 처음 켤 때만 장면 정리 (그 뒤로는 물체가 계속 이어진다)
        {
            conveyorMode = true;
            pickMode = true;
            convRng = new System.Random(seed);
            targetT.gameObject.SetActive(false);
            beltRoot.SetActive(true);
            for (int i = 0; i < 2; i++) boxT[i].gameObject.SetActive(true);
            gripperGO.SetActive(true);
            tipRend.enabled = false;
            // 팔은 컨베이어(벨트·난간·멈춤판)와 상자 벽을 통과한다.  컨베이어는 학습 때 없었고, 연속 작업에서는
            // 팔이 상자 벽 사이에 끼면 관절 목표만 계속 쌓이다 시간초과가 났다.  물체는 그대로 이것들과 부딪친다
            var noArm = new List<Collider>(beltRoot.GetComponentsInChildren<Collider>());
            for (int i = 0; i < 2; i++) noArm.AddRange(boxT[i].GetComponentsInChildren<Collider>());
            foreach (Collider bc in noArm)
                foreach (Collider c in armCols) Physics.IgnoreCollision(bc, c, true);
            foreach (Rigidbody rb in pool) rb.gameObject.SetActive(false);
            held = false;
            sorted = missorted = 0;
            nextSpawn = 0;
            lastColor = -1;
            colorStreak = 0;
            spawnTimer = spawnInterval;
        }
        else
        {
            int b = ObjInBox();                  // 직전 물체 결과 집계 (화면 표시용)
            if (b == objColor) sorted++;
            else if (b >= 0) missorted++;
        }
        ApplyGrip(false);                        // 들고 있던 것은 놓고
        // 팔은 정면을 보고 똑바로 선 자세로 복귀 (순간이동 없이 관절 구동으로).  v2 는 무작위 자세로 학습했지만
        // 직전 상자 쪽을 향한 채 시작하면 가끔 헤맸다 (빨강 물체에서 약 10%).  정면 자세에서는 매번 성공했다
        for (int i = 0; i < 3; i++) { tgtDeg[i] = 0f; SetDriveTarget(i, 0f); }
        waiting = true;
        waitTicks = 0;
        stepping = true;
    }

    // ------------------------------------------------------------------ 규칙
    void SetDriveTarget(int i, float deg)
    {
        ArticulationDrive d = joint[i].xDrive;
        d.target = deg;
        joint[i].xDrive = d;
    }

    // 집게: 닫는 순간 물체가 가까이 있으면 잡고, 열면 놓는다 (놓인 물체는 중력으로 떨어짐)
    void ApplyGrip(bool close)
    {
        gripClosed = close;
        if (close && !held && (tipT.position - objT.position).magnitude <= curGrabDist)
        {
            held = true;
            objRb.isKinematic = true;
            objCol.enabled = !conveyorMode;      // 컨베이어: 들고 가는 물체가 줄 선 물체를 쳐서 날리지 않게 (학습 때는 물체가 하나뿐)
            FollowTip();
        }
        else if (!close && held)
        {
            held = false;
            objCol.enabled = true;
            objRb.isKinematic = false;
            objRb.velocity = Vector3.zero;
            objRb.angularVelocity = Vector3.zero;
        }
        // 손가락 모양은 '닫기 명령'이 아니라 '실제로 잡았는지'를 따른다.  정책은 물체에 닿기 전에 미리 닫는 경우가
        // 많아서, 명령대로 그리면 닫힌 손가락이 물체를 뚫고 들어가 보인다.  열려 있으면 기울어진 큐브도 손가락 사이에 들어간다
        SetFingers(held);
    }

    void ResetEpisode(int seed)
    {
        System.Random rng = new System.Random(seed);

        // 관절을 0 도(팔을 위로 쭉 편 자세)로 되돌린다
        for (int i = 0; i < 3; i++)
        {
            tgtDeg[i] = 0f;
            joint[i].jointPosition = new ArticulationReducedSpace(0f);
            joint[i].jointVelocity = new ArticulationReducedSpace(0f);
            SetDriveTarget(i, 0f);
        }

        targetT.gameObject.SetActive(!pickMode);
        for (int i = 0; i < 2; i++) boxT[i].gameObject.SetActive(pickMode);
        gripperGO.SetActive(pickMode);
        tipRend.enabled = !pickMode;                                     // 도달 과제에서만 노란 손끝 구를 보여 준다
        beltRoot.SetActive(false);
        foreach (Rigidbody rb in pool) rb.gameObject.SetActive(false);
        SetCurrent(0);
        waiting = false;
        held = false;
        gripClosed = false;
        SetFingers(false);
        if (pickMode) ResetPick(rng);
        else ResetReach(rng);

        // 자세가 반영되도록 몇 틱 진행
        for (int k = 0; k < 5; k++) Physics.Simulate(tickDt);
        stepping = false;
        ticksRemaining = 0;
    }

    void ResetPick(System.Random rng)
    {
        // 물체: 팔 앞쪽(방위각 ±50도), 받침에서 1.3 ~ 2.0 떨어진 바닥 위.  색은 무작위
        double azim = (rng.NextDouble() * 2.0 - 1.0) * 50.0 * Math.PI / 180.0;
        float rho = 1.3f + (float)rng.NextDouble() * 0.7f;
        int color = rng.Next(2);
        Vector3 p = new Vector3(rho * (float)Math.Cos(azim), objSize / 2f + 0.01f, rho * (float)Math.Sin(azim));
        PlaceObject(0, p, color);
        SetCurrent(0);
        if (!randPose) return;

        // 무작위 시작 자세: 손끝·팔꿈치가 높이 1.3 이상이 되는 범위 (바닥·상자와 겹치지 않게)
        float yaw = ((float)rng.NextDouble() * 2f - 1f) * yawLimit;
        float sh = ((float)rng.NextDouble() * 2f - 1f) * 60f;
        float sum = ((float)rng.NextDouble() * 2f - 1f) * 90f;               // 어깨 + 팔꿈치 (아래팔이 수평보다 위)
        float el = Mathf.Clamp(sum - sh, -elbowLimit, elbowLimit);
        float[] deg = { yaw, sh, el };
        for (int i = 0; i < 3; i++)
        {
            tgtDeg[i] = deg[i];
            joint[i].jointPosition = new ArticulationReducedSpace(deg[i] * Mathf.Deg2Rad);
            SetDriveTarget(i, deg[i]);
        }
    }

    void ResetReach(System.Random rng)
    {
        // 목표: 어깨 기준 가로 거리 / 높이 / 방위각을 무작위로 (닿을 수 있는 범위 안에서)
        double azim = (rng.NextDouble() * 2.0 - 1.0) * 135.0 * Math.PI / 180.0;
        float rho = 0.6f + (float)rng.NextDouble() * 1.6f;          // 0.6 ~ 2.2
        float y = 0.5f + (float)rng.NextDouble() * 1.5f;            // 0.5 ~ 2.0
        Vector3 t = new Vector3(rho * (float)Math.Cos(azim), y, rho * (float)Math.Sin(azim));
        if ((t - shoulderPos).magnitude > upperLength + foreLength - 0.25f)
            t = shoulderPos + (t - shoulderPos).normalized * (upperLength + foreLength - 0.25f);
        targetT.position = t;
    }

    // ------------------------------------------------------------------ 상태 보내기
    void SendState()
    {
        ArmState msg = new ArmState();
        msg.joints = new float[3];
        msg.vel = new float[3];
        for (int i = 0; i < 3; i++)
        {
            msg.joints[i] = joint[i].jointPosition[0];
            msg.vel[i] = joint[i].jointVelocity[0];
        }
        Vector3 p = tipT.position;
        Vector3 g = targetT.position;
        msg.tip = new float[] { p.x, p.y, p.z };
        msg.target = new float[] { g.x, g.y, g.z };
        Vector3 o = objT.position, b0 = boxT[0].position, b1 = boxT[1].position;
        msg.obj = new float[] { o.x, o.y, o.z };
        msg.boxes = new float[] { b0.x, b0.y, b0.z, b1.x, b1.y, b1.z };
        msg.held = held ? 1 : 0;
        msg.grip = gripClosed ? 1 : 0;
        msg.color = objColor;
        msg.in_box = ObjInBox();
        if (!connected) return;
        try { writer.WriteLine(JsonUtility.ToJson(msg)); }
        catch (Exception e) { Debug.LogWarning("[Arm] 전송 실패: " + e.Message); }
    }

    void OnGUI()
    {
        if (tipT == null || targetT == null) return;
        if (conveyorMode)
            GUI.Label(new Rect(8, 28, 600, 22), "컨베이어  |  분류 성공 " + sorted + "  |  잘못 분류 " + missorted + (waiting ? "  |  다음 물체 기다리는 중 (" + waitReason + ")" : ""));
        if (pickMode)
        {
            float dg = (objT.position - boxT[objColor].position).magnitude;
            GUI.Label(new Rect(8, 6, 600, 22), "물체 " + (objColor == 0 ? "빨강" : "파랑") + "  |  집게 " + (gripClosed ? "닫힘" : "열림")
                + "  |  " + (held ? "잡고 있음" : "놓음") + "  |  같은 색 상자까지 " + dg.ToString("F2") + "  |  상자 " + ObjInBox());
            return;
        }
        float d = (tipT.position - targetT.position).magnitude;
        GUI.Label(new Rect(8, 6, 460, 22), "손끝-목표 거리 " + d.ToString("F2") + "   (성공 기준 " + successDist.ToString("F2") + ")");
    }

    void OnDestroy()
    {
        running = false;
        try { if (client != null) client.Close(); } catch (Exception) { }
        client = null;
        if (settingsSaved)
        {
            Physics.gravity = prevGravity;
#if UNITY_2022_2_OR_NEWER
            Physics.simulationMode = prevSimMode;
#else
            Physics.autoSimulation = prevAutoSim;
#endif
            QualitySettings.vSyncCount = prevVSync;
            Application.targetFrameRate = prevTargetFps;
        }
    }
}
