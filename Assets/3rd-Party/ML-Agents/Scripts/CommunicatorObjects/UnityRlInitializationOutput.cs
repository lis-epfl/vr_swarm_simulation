// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: mlagents/envs/communicator_objects/unity_rl_initialization_output.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace MLAgents.CommunicatorObjects {

  /// <summary>Holder for reflection information generated from mlagents/envs/communicator_objects/unity_rl_initialization_output.proto</summary>
  public static partial class UnityRlInitializationOutputReflection {

    #region Descriptor
    /// <summary>File descriptor for mlagents/envs/communicator_objects/unity_rl_initialization_output.proto</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static UnityRlInitializationOutputReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "CkdtbGFnZW50cy9lbnZzL2NvbW11bmljYXRvcl9vYmplY3RzL3VuaXR5X3Js",
            "X2luaXRpYWxpemF0aW9uX291dHB1dC5wcm90bxIUY29tbXVuaWNhdG9yX29i",
            "amVjdHMaP21sYWdlbnRzL2VudnMvY29tbXVuaWNhdG9yX29iamVjdHMvYnJh",
            "aW5fcGFyYW1ldGVyc19wcm90by5wcm90bxpFbWxhZ2VudHMvZW52cy9jb21t",
            "dW5pY2F0b3Jfb2JqZWN0cy9lbnZpcm9ubWVudF9wYXJhbWV0ZXJzX3Byb3Rv",
            "LnByb3RvIuYBChtVbml0eVJMSW5pdGlhbGl6YXRpb25PdXRwdXQSDAoEbmFt",
            "ZRgBIAEoCRIPCgd2ZXJzaW9uGAIgASgJEhAKCGxvZ19wYXRoGAMgASgJEkQK",
            "EGJyYWluX3BhcmFtZXRlcnMYBSADKAsyKi5jb21tdW5pY2F0b3Jfb2JqZWN0",
            "cy5CcmFpblBhcmFtZXRlcnNQcm90bxJQChZlbnZpcm9ubWVudF9wYXJhbWV0",
            "ZXJzGAYgASgLMjAuY29tbXVuaWNhdG9yX29iamVjdHMuRW52aXJvbm1lbnRQ",
            "YXJhbWV0ZXJzUHJvdG9CH6oCHE1MQWdlbnRzLkNvbW11bmljYXRvck9iamVj",
            "dHNiBnByb3RvMw=="));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { global::MLAgents.CommunicatorObjects.BrainParametersProtoReflection.Descriptor, global::MLAgents.CommunicatorObjects.EnvironmentParametersProtoReflection.Descriptor, },
          new pbr::GeneratedClrTypeInfo(null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::MLAgents.CommunicatorObjects.UnityRLInitializationOutput), global::MLAgents.CommunicatorObjects.UnityRLInitializationOutput.Parser, new[]{ "Name", "Version", "LogPath", "BrainParameters", "EnvironmentParameters" }, null, null, null)
          }));
    }
    #endregion

  }
  #region Messages
  /// <summary>
  /// The request message containing the academy's parameters.
  /// </summary>
  public sealed partial class UnityRLInitializationOutput : pb::IMessage<UnityRLInitializationOutput> {
    private static readonly pb::MessageParser<UnityRLInitializationOutput> _parser = new pb::MessageParser<UnityRLInitializationOutput>(() => new UnityRLInitializationOutput());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<UnityRLInitializationOutput> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::MLAgents.CommunicatorObjects.UnityRlInitializationOutputReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public UnityRLInitializationOutput() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public UnityRLInitializationOutput(UnityRLInitializationOutput other) : this() {
      name_ = other.name_;
      version_ = other.version_;
      logPath_ = other.logPath_;
      brainParameters_ = other.brainParameters_.Clone();
      environmentParameters_ = other.environmentParameters_ != null ? other.environmentParameters_.Clone() : null;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public UnityRLInitializationOutput Clone() {
      return new UnityRLInitializationOutput(this);
    }

    /// <summary>Field number for the "name" field.</summary>
    public const int NameFieldNumber = 1;
    private string name_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string Name {
      get { return name_; }
      set {
        name_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "version" field.</summary>
    public const int VersionFieldNumber = 2;
    private string version_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string Version {
      get { return version_; }
      set {
        version_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "log_path" field.</summary>
    public const int LogPathFieldNumber = 3;
    private string logPath_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string LogPath {
      get { return logPath_; }
      set {
        logPath_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    /// <summary>Field number for the "brain_parameters" field.</summary>
    public const int BrainParametersFieldNumber = 5;
    private static readonly pb::FieldCodec<global::MLAgents.CommunicatorObjects.BrainParametersProto> _repeated_brainParameters_codec
        = pb::FieldCodec.ForMessage(42, global::MLAgents.CommunicatorObjects.BrainParametersProto.Parser);
    private readonly pbc::RepeatedField<global::MLAgents.CommunicatorObjects.BrainParametersProto> brainParameters_ = new pbc::RepeatedField<global::MLAgents.CommunicatorObjects.BrainParametersProto>();
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public pbc::RepeatedField<global::MLAgents.CommunicatorObjects.BrainParametersProto> BrainParameters {
      get { return brainParameters_; }
    }

    /// <summary>Field number for the "environment_parameters" field.</summary>
    public const int EnvironmentParametersFieldNumber = 6;
    private global::MLAgents.CommunicatorObjects.EnvironmentParametersProto environmentParameters_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public global::MLAgents.CommunicatorObjects.EnvironmentParametersProto EnvironmentParameters {
      get { return environmentParameters_; }
      set {
        environmentParameters_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as UnityRLInitializationOutput);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(UnityRLInitializationOutput other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (Name != other.Name) return false;
      if (Version != other.Version) return false;
      if (LogPath != other.LogPath) return false;
      if(!brainParameters_.Equals(other.brainParameters_)) return false;
      if (!object.Equals(EnvironmentParameters, other.EnvironmentParameters)) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (Name.Length != 0) hash ^= Name.GetHashCode();
      if (Version.Length != 0) hash ^= Version.GetHashCode();
      if (LogPath.Length != 0) hash ^= LogPath.GetHashCode();
      hash ^= brainParameters_.GetHashCode();
      if (environmentParameters_ != null) hash ^= EnvironmentParameters.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (Name.Length != 0) {
        output.WriteRawTag(10);
        output.WriteString(Name);
      }
      if (Version.Length != 0) {
        output.WriteRawTag(18);
        output.WriteString(Version);
      }
      if (LogPath.Length != 0) {
        output.WriteRawTag(26);
        output.WriteString(LogPath);
      }
      brainParameters_.WriteTo(output, _repeated_brainParameters_codec);
      if (environmentParameters_ != null) {
        output.WriteRawTag(50);
        output.WriteMessage(EnvironmentParameters);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (Name.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(Name);
      }
      if (Version.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(Version);
      }
      if (LogPath.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(LogPath);
      }
      size += brainParameters_.CalculateSize(_repeated_brainParameters_codec);
      if (environmentParameters_ != null) {
        size += 1 + pb::CodedOutputStream.ComputeMessageSize(EnvironmentParameters);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(UnityRLInitializationOutput other) {
      if (other == null) {
        return;
      }
      if (other.Name.Length != 0) {
        Name = other.Name;
      }
      if (other.Version.Length != 0) {
        Version = other.Version;
      }
      if (other.LogPath.Length != 0) {
        LogPath = other.LogPath;
      }
      brainParameters_.Add(other.brainParameters_);
      if (other.environmentParameters_ != null) {
        if (environmentParameters_ == null) {
          environmentParameters_ = new global::MLAgents.CommunicatorObjects.EnvironmentParametersProto();
        }
        EnvironmentParameters.MergeFrom(other.EnvironmentParameters);
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10: {
            Name = input.ReadString();
            break;
          }
          case 18: {
            Version = input.ReadString();
            break;
          }
          case 26: {
            LogPath = input.ReadString();
            break;
          }
          case 42: {
            brainParameters_.AddEntriesFrom(input, _repeated_brainParameters_codec);
            break;
          }
          case 50: {
            if (environmentParameters_ == null) {
              environmentParameters_ = new global::MLAgents.CommunicatorObjects.EnvironmentParametersProto();
            }
            input.ReadMessage(environmentParameters_);
            break;
          }
        }
      }
    }

  }

  #endregion

}

#endregion Designer generated code
