//! `rbuilder_*` state-access RPC extension.
//!
//! Serves the historical-state access methods that flashbots rbuilder's
//! IPC/HTTP state providers expect from an "extended" node
//! (`crates/rbuilder/src/provider/ipc_state_provider.rs`):
//!
//! - `rbuilder_getAccount(address, blockId)` — basic account info at a
//!   historical block (`None` when the account does not exist).
//! - `rbuilder_getCodeByHash(codeHash)` — contract bytecode by its hash.
//! - `rbuilder_getBlockHash(number)` — canonical block hash by number.
//! - `rbuilder_calculateStateRoot(blockId, accountDiffs)` — state root of the
//!   historical state overlaid with the builder's pending changes (the
//!   embedded-EVM root-hash path).
//!
//! All methods are read-only, so the namespace is registered unconditionally
//! like `parlia`.

use alloy_primitives::{map::HashMap, Address, Bytes, B256, U256};
use jsonrpsee::core::RpcResult;
use jsonrpsee::proc_macros::rpc;
use reth_provider::{BlockNumReader, StateProviderFactory};
use reth_storage_api::BlockIdReader;
use reth_trie::HashedPostState;
use serde::{Deserialize, Serialize};

/// One account's pending changes as sent by the builder
/// (rbuilder `AccountDiff`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbuilderAccountDiff {
    pub nonce: Option<U256>,
    pub balance: Option<U256>,
    pub self_destructed: bool,
    pub changed_slots: HashMap<U256, U256>,
    pub code_hash: Option<B256>,
}

/// Basic account info as returned to rbuilder (`rbuilder::AccountState`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbuilderAccountState {
    pub nonce: U256,
    pub balance: U256,
    pub code_hash: B256,
}

/// Keccak256 of empty bytecode — rbuilder treats this as "no code".
const KECCAK_EMPTY: B256 = B256::new([
    0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03,
    0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85,
    0xa4, 0x70,
]);

#[rpc(server, namespace = "rbuilder")]
pub trait RbuilderStateApi {
    /// Basic account information at a historical block.
    #[method(name = "getAccount")]
    async fn get_account(
        &self,
        address: Address,
        block_id: alloy_rpc_types::BlockId,
    ) -> RpcResult<Option<RbuilderAccountState>>;

    /// Contract bytecode by its keccak256 hash.
    #[method(name = "getCodeByHash")]
    async fn get_code_by_hash(&self, code_hash: B256) -> RpcResult<Option<Bytes>>;

    /// Canonical block hash by block number.
    #[method(name = "getBlockHash")]
    async fn get_block_hash(
        &self,
        number: alloy_rpc_types::BlockNumberOrTag,
    ) -> RpcResult<Option<B256>>;

    /// State root of the historical state overlaid with the builder's pending
    /// account changes.
    #[method(name = "calculateStateRoot")]
    async fn calculate_state_root(
        &self,
        block_id: alloy_rpc_types::BlockId,
        account_diffs: HashMap<Address, RbuilderAccountDiff>,
    ) -> RpcResult<B256>;
}

/// Implementation backed by the node's blockchain provider.
#[derive(Clone)]
pub struct RbuilderStateApiImpl<Provider> {
    provider: Provider,
}

impl<Provider> RbuilderStateApiImpl<Provider> {
    pub fn new(provider: Provider) -> Self {
        Self { provider }
    }
}

fn err(e: impl std::fmt::Display) -> jsonrpsee::types::ErrorObjectOwned {
    jsonrpsee::types::ErrorObject::owned(-32050, e.to_string(), None::<()>)
}

/// Resolve a [`BlockId`] into a historical state provider. rbuilder always
/// sends the parent block **hash**; numeric ids are also accepted.
fn resolve_history_provider<Provider>(
    provider: &Provider,
    block_id: alloy_rpc_types::BlockId,
) -> Result<reth_storage_api::StateProviderBox, jsonrpsee::types::ErrorObjectOwned>
where
    Provider: StateProviderFactory + BlockIdReader,
{
    let block_hash = match block_id {
        alloy_rpc_types::BlockId::Hash(h) => Some(h.block_hash),
        alloy_rpc_types::BlockId::Number(n) => {
            let n = n.as_number().ok_or_else(|| {
                jsonrpsee::types::ErrorObject::owned(
                    -32051,
                    "pending/safe/finalized tags are not supported",
                    None::<()>,
                )
            })?;
            provider.block_hash(n).map_err(|e| err(e))?
        }
        _ => {
            return Err(jsonrpsee::types::ErrorObject::owned(
                -32051,
                "unsupported block id",
                None::<()>,
            ))
        }
    };
    let hash = block_hash
        .ok_or_else(|| jsonrpsee::types::ErrorObject::owned(-32050, "block not found", None::<()>))?;
    provider.history_by_block_hash(hash).map_err(|e| err(e))
}

#[async_trait::async_trait]
impl<Provider> RbuilderStateApiServer for RbuilderStateApiImpl<Provider>
where
    Provider: StateProviderFactory + BlockNumReader + BlockIdReader + Send + Sync + 'static,
{
    async fn get_account(
        &self,
        address: Address,
        block_id: alloy_rpc_types::BlockId,
    ) -> RpcResult<Option<RbuilderAccountState>> {
        let provider = resolve_history_provider(&self.provider, block_id)?;
        let account = provider.basic_account(&address).map_err(|e| err(e))?;
        Ok(account.map(|a| RbuilderAccountState {
            nonce: U256::from(a.nonce),
            balance: a.balance,
            code_hash: a.bytecode_hash.unwrap_or(KECCAK_EMPTY),
        }))
    }

    async fn get_code_by_hash(&self, code_hash: B256) -> RpcResult<Option<Bytes>> {
        if code_hash == KECCAK_EMPTY || code_hash == B256::ZERO {
            return Ok(None);
        }
        let latest = self.provider.best_block_number().map_err(|e| err(e))?;
        let hash = self
            .provider
            .block_hash(latest)
            .map_err(|e| err(e))?
            .ok_or_else(|| jsonrpsee::types::ErrorObject::owned(-32050, "head block not found", None::<()>))?;
        let provider = self.provider.history_by_block_hash(hash).map_err(|e| err(e))?;
        let code = provider.bytecode_by_hash(&code_hash).map_err(|e| err(e))?;
        Ok(code.map(|b| Bytes::from(b.original_bytes())))
    }

    async fn get_block_hash(
        &self,
        number: alloy_rpc_types::BlockNumberOrTag,
    ) -> RpcResult<Option<B256>> {
        let block_number = number.as_number().ok_or_else(|| {
            jsonrpsee::types::ErrorObject::owned(
                -32051,
                "only numeric block numbers are supported",
                None::<()>,
            )
        })?;
        let hash = self.provider.block_hash(block_number).map_err(|e| err(e))?;
        Ok(hash)
    }

    async fn calculate_state_root(
        &self,
        block_id: alloy_rpc_types::BlockId,
        account_diffs: HashMap<Address, RbuilderAccountDiff>,
    ) -> RpcResult<B256> {
        let state_provider = resolve_history_provider(&self.provider, block_id)?;

        let mut hashed = HashedPostState::default();
        for (address, diff) in account_diffs {
            let hashed_address = alloy_primitives::keccak256(address);

            if diff.self_destructed {
                // Account destroyed: `None` marks the deletion in the trie overlay.
                hashed.accounts.insert(hashed_address, None);
                hashed.storages.insert(
                    hashed_address,
                    reth_trie_common::HashedStorage::new(true),
                );
                continue;
            }

            // Base account from the historical state; apply the builder's diff
            // on top (rbuilder sends only the changed fields).
            let mut info = state_provider
                .basic_account(&address)
                .map_err(|e| err(e))?
                .unwrap_or_default();

            if let Some(balance) = diff.balance {
                info.balance = balance;
            }
            if let Some(nonce) = diff.nonce {
                info.nonce = nonce.to::<u64>();
            }
            if let Some(code_hash) = diff.code_hash {
                info.bytecode_hash = Some(code_hash);
            }
            hashed.accounts.insert(hashed_address, Some(info));

            if !diff.changed_slots.is_empty() {
                let storage = hashed.storages.entry(hashed_address).or_default();
                for (slot, value) in diff.changed_slots {
                    storage
                        .storage
                        .insert(alloy_primitives::keccak256(B256::new(slot.to_be_bytes())), value);
                }
            }
        }

        let root = state_provider.state_root(hashed).map_err(|e| err(e))?;
        Ok(root)
    }
}
