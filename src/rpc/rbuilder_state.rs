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
use reth_provider::{BlockNumReader, ProviderError, StateProviderFactory};
use reth_storage_api::BlockIdReader;
use reth_trie::HashedPostState;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

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

/// Error code of a read that kept running into an on-disk unwind (see
/// [`is_truncated_read`]); rbuilder takes it as transient.
pub const TRUNCATED_READ_CODE: i32 = -32052;

/// How long a request keeps retrying reads that ran into an on-disk unwind. rbuilder gives
/// a state-root call 300 ms; the unwinds seen on a privnet truncate for about 80 ms.
const TRUNCATED_READ_BUDGET: Duration = Duration::from_millis(150);
const TRUNCATED_READ_BACKOFF: Duration = Duration::from_millis(25);

fn err(e: ProviderError) -> jsonrpsee::types::ErrorObjectOwned {
    let code = if is_truncated_read(&e) { TRUNCATED_READ_CODE } else { -32050 };
    jsonrpsee::types::ErrorObject::owned(code, e.to_string(), None::<()>)
}

/// Whether a read ran past the end of a static file.
///
/// Reth unwinds a persisted block (a reorg of the head it just persisted) by truncating the
/// changeset files in place, while readers opened before keep their old length: a
/// historical read at the parent then fails with `UnexpectedEof` ("failed to fill whole
/// buffer"). The same read succeeds once the unwind is done.
fn is_truncated_read(error: &ProviderError) -> bool {
    let mut source = error.as_other().map(|e| e as &(dyn std::error::Error + 'static));
    while let Some(error) = source {
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::UnexpectedEof)
        {
            return true;
        }
        source = error.source();
    }
    false
}

/// Runs `read`, again while it fails on a truncated read and the budget lasts.
async fn retry_truncated_reads<T>(mut read: impl FnMut() -> RpcResult<T>) -> RpcResult<T> {
    let start = Instant::now();
    loop {
        match read() {
            Err(error)
                if error.code() == TRUNCATED_READ_CODE &&
                    start.elapsed() + TRUNCATED_READ_BACKOFF <= TRUNCATED_READ_BUDGET =>
            {
                tracing::debug!(
                    target: "rpc::rbuilder",
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "state read ran into an on-disk unwind, retrying"
                );
                tokio::time::sleep(TRUNCATED_READ_BACKOFF).await;
            }
            result => return result,
        }
    }
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
            provider.block_hash(n).map_err(err)?
        }
    };
    let hash = block_hash
        .ok_or_else(|| jsonrpsee::types::ErrorObject::owned(-32050, "block not found", None::<()>))?;
    provider.history_by_block_hash(hash).map_err(err)
}

impl<Provider> RbuilderStateApiImpl<Provider>
where
    Provider: StateProviderFactory + BlockNumReader + BlockIdReader,
{
    fn account(
        &self,
        address: Address,
        block_id: alloy_rpc_types::BlockId,
    ) -> RpcResult<Option<RbuilderAccountState>> {
        let provider = resolve_history_provider(&self.provider, block_id)?;
        let account = provider.basic_account(&address).map_err(err)?;
        Ok(account.map(|a| RbuilderAccountState {
            nonce: U256::from(a.nonce),
            balance: a.balance,
            code_hash: a.bytecode_hash.unwrap_or(KECCAK_EMPTY),
        }))
    }

    fn code_by_hash(&self, code_hash: B256) -> RpcResult<Option<Bytes>> {
        let latest = self.provider.best_block_number().map_err(err)?;
        let hash = self
            .provider
            .block_hash(latest)
            .map_err(err)?
            .ok_or_else(|| jsonrpsee::types::ErrorObject::owned(-32050, "head block not found", None::<()>))?;
        let provider = self.provider.history_by_block_hash(hash).map_err(err)?;
        let code = provider.bytecode_by_hash(&code_hash).map_err(err)?;
        Ok(code.map(|b| b.original_bytes()))
    }

    fn state_root(
        &self,
        block_id: alloy_rpc_types::BlockId,
        account_diffs: &HashMap<Address, RbuilderAccountDiff>,
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
                .basic_account(address)
                .map_err(err)?
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
                for (slot, value) in &diff.changed_slots {
                    storage
                        .storage
                        .insert(alloy_primitives::keccak256(B256::new(slot.to_be_bytes())), *value);
                }
            }
        }

        state_provider.state_root(hashed).map_err(err)
    }
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
        retry_truncated_reads(|| self.account(address, block_id)).await
    }

    async fn get_code_by_hash(&self, code_hash: B256) -> RpcResult<Option<Bytes>> {
        if code_hash == KECCAK_EMPTY || code_hash == B256::ZERO {
            return Ok(None);
        }
        retry_truncated_reads(|| self.code_by_hash(code_hash)).await
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
        let hash = self.provider.block_hash(block_number).map_err(err)?;
        Ok(hash)
    }

    async fn calculate_state_root(
        &self,
        block_id: alloy_rpc_types::BlockId,
        account_diffs: HashMap<Address, RbuilderAccountDiff>,
    ) -> RpcResult<B256> {
        retry_truncated_reads(|| self.state_root(block_id, &account_diffs)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eof() -> ProviderError {
        ProviderError::other(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "failed to fill whole buffer",
        ))
    }

    #[test]
    fn a_read_past_a_truncated_file_has_its_own_code() {
        assert!(is_truncated_read(&eof()));
        assert_eq!(err(eof()).code(), TRUNCATED_READ_CODE);

        let other = ProviderError::other(std::io::Error::other("disk on fire"));
        assert!(!is_truncated_read(&other));
        assert_eq!(err(other).code(), -32050);
        assert_eq!(err(ProviderError::BestBlockNotFound).code(), -32050);
    }

    #[tokio::test]
    async fn a_truncated_read_is_retried_until_the_unwind_is_done() {
        let mut calls = 0;
        let result = retry_truncated_reads(|| {
            calls += 1;
            if calls < 3 { Err(err(eof())) } else { Ok(calls) }
        })
        .await;
        assert_eq!(result.unwrap(), 3);
    }

    #[tokio::test]
    async fn retries_stop_within_the_budget() {
        let start = Instant::now();
        let mut calls = 0;
        let result: RpcResult<()> = retry_truncated_reads(|| {
            calls += 1;
            Err(err(eof()))
        })
        .await;
        assert_eq!(result.unwrap_err().code(), TRUNCATED_READ_CODE);
        assert!(start.elapsed() <= TRUNCATED_READ_BUDGET + TRUNCATED_READ_BACKOFF);
        assert!(calls > 1);

        // Any other failure is final at once.
        let mut calls = 0;
        let _: RpcResult<()> = retry_truncated_reads(|| {
            calls += 1;
            Err(err(ProviderError::BestBlockNotFound))
        })
        .await;
        assert_eq!(calls, 1);
    }
}
